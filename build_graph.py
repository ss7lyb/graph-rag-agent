from dotenv import load_dotenv
from model.get_models import get_llm_model
from config.prompt import system_template, human_template
from config.settings import entity_types, relationship_types,theme, FILES_DIR, CHUNK_SIZE, OVERLAP
from processor.file_reader import FileReader
from processor.text_chunker import ChineseTextChunker
from graph.struct_builder import GraphStructureBuilder
from graph.entity_extractor import EntityRelationExtractor
from graph.graph_writer import GraphWriter

load_dotenv()

# 读入文件------------------------------------------------------------------
fr = FileReader(FILES_DIR)
# 调用函数并打印结果
file_contents = fr.read_txt_files()
for file_name, content in file_contents:
    print("文件名:", file_name)


# 文本分块----------------------------------------------------------------------
# 使用自定义函数进行分块
chunker = ChineseTextChunker(CHUNK_SIZE, OVERLAP)

for file_content in file_contents:
    print("文件名:", file_content[0])
    chunks = chunker.chunk_text(file_content[1])
    file_content.append(chunks)
    
# 打印分块结果
for file_content in file_contents:
    print(f"File: {file_content[0]} Chunks: {len(file_content[2])}")
    for i, chunk in enumerate(file_content[2]):
        print(f"Chunk {i+1}: {len(chunk)} tokens.")

# 打印分块内容
print(''.join(file_contents[0][2][0]))


# 在Neo4j中创建文档与Chunk的图结构----------------------------------------------
# 首先实例化 GraphStructureBuilder
struct_builder = GraphStructureBuilder()

graph = struct_builder.graph 

# 清空数据库
struct_builder.clear_database()

# 创建Document节点
for file_content in file_contents:
    struct_builder.create_document(
        type="local",
        uri=str(FILES_DIR),
        file_name=file_content[0],
        domain=theme
    )

# 创建Chunk结点并建立Chunk之间及与Document之间的关系
for file_content in file_contents:
    file_name = file_content[0]
    chunks = file_content[2]
    result = struct_builder.create_relation_between_chunks(file_name, chunks)
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
# 创建实体关系提取器实例
extractor = EntityRelationExtractor(
    llm=llm,
    system_template=system_template,
    human_template=human_template,
    entity_types=entity_types,
    relationship_types=relationship_types
)

# 处理所有文件内容并提取实体关系
# 这里的 file_contents 将被更新，在每个 file_content 列表中添加提取结果
file_contents = extractor.process_chunks(file_contents)

# 提取的实体关系写入Neo4j-------------------------------------------------------
## 创建 GraphWriter 实例
graph_writer = GraphWriter(graph)

# 处理并写入所有文件的实体关系到Neo4j
graph_writer.process_and_write_graph_documents(file_contents)