import codecs
import hashlib
import logging
import os
from typing import List
import time
import re

from dotenv import load_dotenv
import hanlp
from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

from get_models import get_embeddings_model, get_llm_model
from prompt import system_template, human_template, entity_types, relationship_types,theme

load_dotenv()

# 文件目录
directory_path = './files'

# 读入文件------------------------------------------------------------------
def read_txt_files(directory):
    # 存放结果的列表
    results = []
    # 遍历指定目录下的所有文件和文件夹
    for filename in os.listdir(directory):
        # 检查文件扩展名是否为.txt
        if filename.endswith(".txt"):
            # 构建完整的文件路径
            file_path = os.path.join(directory, filename)
            # 打开并读取文件内容
            with codecs.open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # 将文件名和内容以列表形式添加到结果列表
            results.append([filename, content])
    
    return results


# 调用函数并打印结果
file_contents = read_txt_files(directory_path)
for file_name, content in file_contents:
    print("文件名:", file_name)


# 文本分块----------------------------------------------------------------------
# 单任务模型，分词，token的计数是计算词，包括标点符号
tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)

# 划分段落
def split_into_paragraphs(text):
    return text.split('\n')

# 判断token是否为句子结束符
def is_sentence_end(token):
    return token in ['。', '！', '？']

# 向后查找到句子结束符，用于动态调整chunk划分以保证chunk以完整的句子结束
def find_sentence_boundary_forward(tokens, chunk_size):
    end = len(tokens)  # 默认的end值设置为tokens的长度
    for i in range(chunk_size, len(tokens)):  # 从chunk_size开始向后查找
        if is_sentence_end(tokens[i]):
            end = i + 1  # 包含句尾符号
            break
    return end  

# 从位置start开始向前寻找上一句的句子结束符，以保证分块重叠的部分从一个完整的句子开始
def find_sentence_boundary_backward(tokens, start):
    for i in range(start - 1, -1, -1):
        if is_sentence_end(tokens[i]):
            return i + 1  # 包含句尾符号
    return 0  # 找不到
  
# 文本分块，文本块的参考大小为chunk_size，文本块之间重叠部分的参考大小为overlap
# 为了保证文本块之间重叠的部分及文本块末尾截断的部分都是完整的句子，
# 文本块的大小和重叠部分的大小都是根据当前文本块的内容动态调整的，是浮动的值
def chunk_text(text, chunk_size=300, overlap=50):
    if chunk_size <= overlap:  # 参数检查
        raise ValueError("chunk_size must be greater than overlap.")
    # 先划分为段落，段落保存了语义上的信息，整个段落去处理
    paragraphs = split_into_paragraphs(text)
    chunks = []
    buffer = []
    # 逐个段落处理
    i = 0
    while i < len(paragraphs):
        # 注满buffer，直到大于chunk_szie，整个段落读入，段落保存了语义上的信息
        while len(buffer) < chunk_size and i < len(paragraphs):
            tokens = tokenizer(paragraphs[i])
            buffer.extend(tokens)
            i += 1
        # 当前buffer分块
        while len(buffer) >= chunk_size:
            # 保证从完整的句子处截断
            end = find_sentence_boundary_forward(buffer, chunk_size)
            chunk = buffer[:end]
            chunks.append(chunk)  # 保留token的状态以便后面计数
            # 保证重叠的部分从完整的句子开始。
            start_next = find_sentence_boundary_backward(buffer, end - overlap)
            if start_next==0:  # 找不到了上一句的句子结束符，调整重叠范围再找一次
                start_next = find_sentence_boundary_backward(buffer, end-1)
            if start_next==0:  # 真的找不到，放弃块首的完整句子重叠
                start_next = end - overlap
            buffer=buffer[start_next:]
        
    if buffer:  # 如果缓冲区还有剩余的token
        # 检查一下剩余部分是否已经包含在最后一个分块之中，它只是留作块间重叠
        last_chunk = chunks[len(chunks)-1]
        rest = ''.join(buffer)
        temp = ''.join(last_chunk[len(last_chunk)-len(rest):])
        if temp!=rest:   # 如果不是留作重叠，则是最后的一个分块
            chunks.append(buffer)
    
    return chunks


# 使用自定义函数进行分块
for file_content in file_contents:
    print("文件名:", file_content[0])
    chunks = chunk_text(file_content[1], chunk_size=500, overlap=50)
    file_content.append(chunks)
    
# 打印分块结果
for file_content in file_contents:
    print(f"File: {file_content[0]} Chunks: {len(file_content[2])}")
    for i, chunk in enumerate(file_content[2]):
        print(f"Chunk {i+1}: {len(chunk)} tokens.")

# 打印分块内容
print(''.join(file_contents[0][2][0]))


# 在Neo4j中创建文档与Chunk的图结构----------------------------------------------
# 连接neo4j
graph = Neo4jGraph(refresh_schema=True)

# 创建Document结点，与Chunk之间按属性名fileName匹配。
def create_Document(graph,type,uri,file_name, domain):
    query = """
    MERGE(d:`__Document__` {fileName :$file_name}) SET d.type=$type,
          d.uri=$uri, d.domain=$domain
    RETURN d;
    """
    doc = graph.query(query,{"file_name":file_name,"type":type,"uri":uri,"domain":domain})
    return doc
  
# 创建Document结点
for file_content in file_contents:
    doc = create_Document(graph,"local",directory_path,file_content[0],theme)

#创建Chunk结点并建立Chunk之间及与Document之间的关系
#这个程序直接从Neo4j KG Builder拷贝引用，为了增加tokens属性稍作修改。
#https://github.com/neo4j-labs/llm-graph-builder/blob/main/backend/src/make_relationships.py
def create_relation_between_chunks(graph, file_name, chunks: List)->list:
    logging.info("creating FIRST_CHUNK and NEXT_CHUNK relationships between chunks")
    current_chunk_id = ""
    lst_chunks_including_hash = []
    batch_data = []
    relationships = []
    offset=0
    for i, chunk in enumerate(chunks):
        page_content = ''.join(chunk)
        page_content_sha1 = hashlib.sha1(page_content.encode()) # chunk.page_content.encode()
        previous_chunk_id = current_chunk_id
        current_chunk_id = page_content_sha1.hexdigest()
        position = i + 1 
        if i>0:
            last_page_content = ''.join(chunks[i-1])
            offset += len(last_page_content)  # chunks[i-1].page_content
        if i == 0:
            firstChunk = True
        else:
            firstChunk = False  
        metadata = {"position": position,"length": len(page_content), "content_offset":offset, "tokens":len(chunk)}
        chunk_document = Document(
            page_content=page_content, metadata=metadata
        )
        
        chunk_data = {
            "id": current_chunk_id,
            "pg_content": chunk_document.page_content,
            "position": position,
            "length": chunk_document.metadata["length"],
            "f_name": file_name,
            "previous_id" : previous_chunk_id,
            "content_offset" : offset,
            "tokens" : len(chunk)
        }
        
        batch_data.append(chunk_data)
        
        lst_chunks_including_hash.append({'chunk_id': current_chunk_id, 'chunk_doc': chunk_document})
        
        # create relationships between chunks
        if firstChunk:
            relationships.append({"type": "FIRST_CHUNK", "chunk_id": current_chunk_id})
        else:
            relationships.append({
                "type": "NEXT_CHUNK",
                "previous_chunk_id": previous_chunk_id,  # ID of previous chunk
                "current_chunk_id": current_chunk_id
            })
          
    query_to_create_chunk_and_PART_OF_relation = """
        UNWIND $batch_data AS data
        MERGE (c:`__Chunk__` {id: data.id})
        SET c.text = data.pg_content, c.position = data.position, c.length = data.length, c.fileName=data.f_name, 
            c.content_offset=data.content_offset, c.tokens=data.tokens
        WITH data, c
        MATCH (d:`__Document__` {fileName: data.f_name})
        MERGE (c)-[:PART_OF]->(d)
    """
    graph.query(query_to_create_chunk_and_PART_OF_relation, params={"batch_data": batch_data})
    
    query_to_create_FIRST_relation = """ 
        UNWIND $relationships AS relationship
        MATCH (d:`__Document__` {fileName: $f_name})
        MATCH (c:`__Chunk__` {id: relationship.chunk_id})
        FOREACH(r IN CASE WHEN relationship.type = 'FIRST_CHUNK' THEN [1] ELSE [] END |
                MERGE (d)-[:FIRST_CHUNK]->(c))
        """
    graph.query(query_to_create_FIRST_relation, params={"f_name": file_name, "relationships": relationships})   
    
    query_to_create_NEXT_CHUNK_relation = """ 
        UNWIND $relationships AS relationship
        MATCH (c:`__Chunk__` {id: relationship.current_chunk_id})
        WITH c, relationship
        MATCH (pc:`__Chunk__` {id: relationship.previous_chunk_id})
        FOREACH(r IN CASE WHEN relationship.type = 'NEXT_CHUNK' THEN [1] ELSE [] END |
                MERGE (c)<-[:NEXT_CHUNK]-(pc))
        """
    graph.query(query_to_create_NEXT_CHUNK_relation, params={"relationships": relationships})   
    
    return lst_chunks_including_hash

# 创建Chunk结点并建立Chunk之间及与Document之间的关系
for file_content in file_contents:
    file_name = file_content[0]
    chunks = file_content[2]
    result = create_relation_between_chunks(graph, file_name , chunks)
    file_content.append(result)


# 用LLM在每个文本块中提取实体关系 -------------------------------------
llm = get_llm_model()

# llm_transformer 不支持国产模型
# llm_transformer = LLMGraphTransformer(
#     llm=llm,
#     node_properties=["description"],
#     relationship_properties=["description"]
# )


# def process_text(text: str) -> List[GraphDocument]:
#     doc = Document(page_content=text)
#     return llm_transformer.convert_to_graph_documents([doc])

# 手动实现提取实体
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, MessagesPlaceholder("chat_history"), human_message_prompt]
)

chain = chat_prompt | llm


tuple_delimiter = " : "
record_delimiter = "\n"
completion_delimiter = "\n\n"

chat_history = []

t0 = time.time()
for file_content in file_contents:
    results = []
    for chunk in file_content[2]:
        t1 = time.time()
        input_text = ''.join(chunk)
        answer = chain.invoke({
        "chat_history": chat_history,
        "entity_types": entity_types,
        "relationship_types": relationship_types,
        "tuple_delimiter": tuple_delimiter,
        "record_delimiter": record_delimiter,
        "completion_delimiter": completion_delimiter,
        "input_text": input_text
        })
        t2 = time.time()
        results.append(answer.content)
        print(input_text)
        print("\n")
        print(answer.content)
        print("块耗时：",t2-t1,"秒")
        print("\n")
        
    print("文件耗时：",t2-t0,"秒")
    print("\n\n")
    file_content.append(results)

# 提取的实体关系写入Neo4j-------------------------------------------------------
# 自己写代码由 answer.content生成一个GraphDocument对象
# 每个GraphDocument对象里增加一个metadata属性chunk_id，以便与前面建立的Chunk结点关联

# 将每个块提取的实体关系文本转换为LangChain的GraphDocument对象
def convert_to_graph_document(chunk_id, input_text, result):
    # 提取节点和关系
    node_pattern = re.compile(r'\("entity" : "(.+?)" : "(.+?)" : "(.+?)"\)')
    relationship_pattern = re.compile(r'\("relationship" : "(.+?)" : "(.+?)" : "(.+?)" : "(.+?)" : (.+?)\)')

    nodes = {}
    relationships = []

    # 解析节点
    for match in node_pattern.findall(result):
        node_id, node_type, description = match
        if node_id not in nodes:
            nodes[node_id] = Node(id=node_id, type=node_type, properties={'description': description})

    # 解析并处理关系
    for match in relationship_pattern.findall(result):
        source_id, target_id, type, description, weight = match
        # 确保source节点存在
        if source_id not in nodes:
            nodes[source_id] = Node(id=source_id, type="未知", properties={'description': 'No additional data'})
        # 确保target节点存在
        if target_id not in nodes:
            nodes[target_id] = Node(id=target_id, type="未知", properties={'description': 'No additional data'})
        relationships.append(Relationship(source=nodes[source_id], target=nodes[target_id], type=type,
            properties={"description":description, "weight":float(weight)}))

    # 创建图对象
    graph_document = GraphDocument(
        nodes=list(nodes.values()),
        relationships=relationships,
        # page_content不能为空。
        source=Document(page_content=input_text, metadata={"chunk_id": chunk_id})
    )
    return graph_document

# 构造所有文档所有Chunk的GraphDocument对象
for file_content in file_contents:
    chunks = file_content[3]
    results = file_content[4]
    
    graph_documents = []
    for chunk, result in zip(chunks, results):
        graph_document =  convert_to_graph_document(chunk["chunk_id"] ,chunk["chunk_doc"].page_content, result)
        graph_documents.append(graph_document)
        # print(chunk)
        # print(result)
        # print(graph_document)
        # print("\n\n")
    file_content.append(graph_documents)
    
# 实体关系图写入Neo4j，此时每个Chunk是作为Documet结点创建的
# 后面再根据chunk_id把这个Document结点与相应的Chunk结点合并
for file_content in file_contents:
    # 删除没有识别出实体关系的空的图对象
    graph_documents = []
    for graph_document in file_content[5]:
        if len(graph_document.nodes)>0 or len(graph_document.relationships)>0:
            graph_documents.append(graph_document)
    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True
    )

# 合并Chunk结点与add_graph_documents()创建的相应Document结点，
# 迁移所有的实体关系到Chunk结点，并删除相应的Document结点。
# 完成Document->Chunk->Entity的结构。
def merge_relationship_between_chunk_and_entites(graph: Neo4jGraph, graph_documents_chunk_chunk_Id : list):
    batch_data = []
    logging.info("Create MENTIONS relationship between chunks and entities")
    for graph_doc_chunk_id in graph_documents_chunk_chunk_Id:
        query_data={
            'chunk_id': graph_doc_chunk_id,
        }
        batch_data.append(query_data)

    if batch_data:
        unwind_query = """
          UNWIND $batch_data AS data
          MATCH (c:`__Chunk__` {id: data.chunk_id}), (d:Document{chunk_id:data.chunk_id})
          WITH c, d
          MATCH (d)-[r:MENTIONS]->(e)
          MERGE (c)-[newR:MENTIONS]->(e)
          ON CREATE SET newR += properties(r)
          DETACH DELETE d
                """
        graph.query(unwind_query, params={"batch_data": batch_data})


# 合并块结点与Document结点
for file_content in file_contents:
    graph_documents_chunk_chunk_Id=[]
    for chunk in file_content[3]:
        graph_documents_chunk_chunk_Id.append(chunk["chunk_id"])
    
    merge_relationship_between_chunk_and_entites(graph, graph_documents_chunk_chunk_Id)