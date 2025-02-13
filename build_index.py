import numpy as np
import os
from dotenv import load_dotenv
import re
import ast

from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from graphdatascience import GraphDataScience
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from langchain_core.output_parsers import StrOutputParser

from model.get_models import get_embeddings_model, get_llm_model
from processor.file_reader import FileReader
from config.settings import FILES_DIR
from config.prompt import system_template_build_index, user_template_build_index, community_template

load_dotenv()

# 测试 embedding -----------------------------------------------
# 计算两个向量的余弦相似度
def cosine_similarity(vector1, vector2):
    # 计算向量的点积
    dot_product = np.dot(vector1, vector2)
    # 计算向量的范数（长度）
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    
    # 检查是否存在零向量，避免除以零
    if norm_vector1 == 0 or norm_vector2 == 0:
        raise ValueError("输入向量之一是零向量，无法计算余弦相似度。")
    
    # 计算余弦相似度
    cosine_sim = dot_product / (norm_vector1 * norm_vector2)
    return cosine_sim

embeddings = get_embeddings_model()

# 测试embedding的使用
tests=["沙僧","沙和尚","猪","猪八戒","孙悟空","悟空"]

results = embeddings.embed_documents(tests)

print(cosine_similarity(results[2],results[3]))
print('\n')

# 长文本Embedding
fr = FileReader(FILES_DIR)
file_contents = fr.read_txt_files()
print(len(file_contents[0][1]))
print('\n')

single_vector = embeddings.embed_query(file_contents[0][1])
temp = file_contents[0][1][100:4100]
single_vector2 = embeddings.embed_query(file_contents[0][1][100:4100])
print(cosine_similarity(single_vector,single_vector2))
print('\n')
single_vector3 = embeddings.embed_query(file_contents[0][1][100:2000]+file_contents[0][1][2100:4100])
print(cosine_similarity(single_vector,single_vector3))
print('\n')

# 实体索引 --------------------------------------------
graph = Neo4jGraph(refresh_schema=True)

# 先清空已有索引
graph.query("DROP INDEX entity_embedding IF EXISTS")

llm = get_llm_model()

embeddings = get_embeddings_model()

# 用['id', 'description']来计算实体结点的Embedding。
vector = Neo4jVector.from_existing_graph(
    embeddings,
    node_label='__Entity__',
    text_node_properties=['id', 'description'],
    embedding_node_property='embedding'
)

# 可通过 show indexes 查看索引

# 找出相似的实体 ------------------------------------------------
# GDS连接Neo4j
gds = GraphDataScience(
    os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
)

# 如果有，先清空
gds.graph.drop("entities", failIfMissing=False)

# 用K近邻算法查找embedding相似值在阈值以内的近邻
# 建立所有实体在内存投影的子图，GDS算法都要通过内存投影运行
# G代表了子图的投影
G, result = gds.graph.project(
    "entities",                   #  Graph name
    "__Entity__",                 #  Node projection
    "*",                          #  Relationship projection
    nodeProperties=["embedding"]  #  Configuration parameters
)

# 根据前面对Embedding模型的测试设置相似性阈值
similarity_threshold = 0.90

# 用KNN算法找出Embedding相似的实体，建立SIMILAR连接
gds.knn.mutate(
  G,
  nodeProperties=['embedding'],
  mutateRelationshipType= 'SIMILAR',
  mutateProperty= 'score',
  similarityCutoff=similarity_threshold
)

# 弱连接组件算法（不分方向），从新识别的SIMILAR关系中识别相识的社区，社区编号存放在结点的wcc属性
gds.wcc.write(
    G,
    writeProperty="wcc",
    relationshipTypes=["SIMILAR"]
)

# 为了截图演示，再执行一次KNN写入Neo4j磁盘存储，这一段是为了输出后面的截图，不是必须的。
gds.knn.write(
  G,
  nodeProperties=['embedding'],
  writeRelationshipType= 'SIMILAR',
  writeProperty= 'score',
  similarityCutoff=similarity_threshold
)

# 弱连接组件算法（不分方向），从新识别的SIMILAR关系中识别相识的社区，社区编号存放在结点的wcc属性
gds.wcc.write(
    G,
    writeProperty="wcc",
    relationshipTypes=["SIMILAR"]
)

# 找出潜在的相同实体
word_edit_distance = 3
potential_duplicate_candidates = graph.query(
    """MATCH (e:`__Entity__`)
    WHERE size(e.id) > 1 // longer than 2 characters
    WITH e.wcc AS community, collect(e) AS nodes, count(*) AS count
    WHERE count > 1
    UNWIND nodes AS node
    // Add text distance
    WITH distinct
      [n IN nodes WHERE apoc.text.distance(toLower(node.id), toLower(n.id)) < $distance | n.id] AS intermediate_results
    WHERE size(intermediate_results) > 1
    WITH collect(intermediate_results) AS results
    // combine groups together if they share elements
    UNWIND range(0, size(results)-1, 1) as index
    WITH results, index, results[index] as result
    WITH apoc.coll.sort(reduce(acc = result, index2 IN range(0, size(results)-1, 1) |
            CASE WHEN index <> index2 AND
                size(apoc.coll.intersection(acc, results[index2])) > 0
                THEN apoc.coll.union(acc, results[index2])
                ELSE acc
            END
    )) as combinedResult
    WITH distinct(combinedResult) as combinedResult
    // extra filtering
    WITH collect(combinedResult) as allCombinedResults
    UNWIND range(0, size(allCombinedResults)-1, 1) as combinedResultIndex
    WITH allCombinedResults[combinedResultIndex] as combinedResult, combinedResultIndex, allCombinedResults
    WHERE NOT any(x IN range(0,size(allCombinedResults)-1,1)
        WHERE x <> combinedResultIndex
        AND apoc.coll.containsAll(allCombinedResults[x], combinedResult)
    )
    RETURN combinedResult
    """, params={'distance': word_edit_distance})

print("\n")
print(potential_duplicate_candidates[:5])

# 用LLM的自然语言处理能力决定合并的实体 ------------------------------
# 看看LLM是否支持with_structured_output()
if hasattr(llm, 'with_structured_output'):
    print("This model supports with_structured_output.")
else:
    print("This model does not support with_structured_output.")

# 由LLM来最终决定哪些实体该合并
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template_build_index)
human_message_prompt = HumanMessagePromptTemplate.from_template(user_template_build_index)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, MessagesPlaceholder("chat_history"), human_message_prompt]
)

chain = chat_prompt | llm

# 调用LLM得到可以合并实体的列表
merged_entities=[]
for canditates in potential_duplicate_candidates:
    chat_history = []
    answer = chain.invoke({
        "chat_history": chat_history,
        "entities": canditates
        })
    merged_entities.append(answer.content)
    print(answer.content)

# 合并实体---------------------------------------
# 将每个实体列表文本转换为Python List
# 返回的是二级列表
def convert_to_list(result):
    list_pattern = re.compile(r'\[.*?\]')
    entity_lists = []

    # 解析实体列表
    for match in list_pattern.findall(result):
        # 使用 ast.literal_eval 将字符串解析为实际的Python列表
        try:
            entity_list = ast.literal_eval(match)
            entity_lists.append(entity_list)
        except Exception as e:
            print(f"Error parsing {match}: {e}")
    
    return entity_lists

# 最终可合并的实体列表是一个二级列表，每组可合并的实体一个列表。
results = []
for canditates in merged_entities:
    # 将返回的二级列表展平为一级列表
    temp = convert_to_list(canditates)
    for entities in temp:
        if (len(entities)>0):
            results.append(entities)
  
print(results)

# 合并实体
graph.query("""
UNWIND $data AS candidates
CALL {
  WITH candidates
  MATCH (e:__Entity__) WHERE e.id IN candidates
  RETURN collect(e) AS nodes
}
CALL apoc.refactor.mergeNodes(nodes, {properties: {
    `.*`: 'discard'
}})
YIELD node
RETURN count(*)
""", params={"data": results})

# 处理完毕，删除内存中的子图投影
G.drop()

# 社区发现----------------------------------------
# 微软的GraphRAG论文指出，在提取的实体上建立社区并作社区摘要，可以为全局性问题的回答提供很好的支持。
# 算法1：Leiden算法

# 建立子图投影
gds.graph.drop("communities", failIfMissing=False)

G, result = gds.graph.project(
    "communities",  #  Graph name
    "__Entity__",  #  Node projection
    {
        "_ALL_": {
            "type": "*",
            "orientation": "UNDIRECTED",
            "properties": {"weight": {"property": "*", "aggregation": "COUNT"}},
        }
    },
)
# 等效的Cypher投影语句：
# https://neo4j.com/docs/graph-data-science/2.6/management-ops/graph-creation/graph-project/
# Cypher语句中的MAP不等同于Python的字典，key不需要用引号括起，值才需要。
# CALL gds.graph.project(
#   'communities',                  // 图的名称
#   '__Entity__',                 // 节点投影
#   {
#   _ALL_: {             // 关系类型的标识符，表示所有类型关系的统一配置
#       type: '*',                  // 匹配所有类型的关系
#       orientation: 'UNDIRECTED',  // 将关系视为无向关系
#       properties: {               // 定义关系属性的处理方式
#         weight: {
#           property: '*',          // 匹配所有属性
#           aggregation: 'COUNT'    // 计数聚合；计算有多少条边满足条件
#         }
#       }
#     }
#   }
# );

wcc = gds.wcc.stats(G)
print(f"Component count: {wcc['componentCount']}")
print(f"Component distribution: {wcc['componentDistribution']}")

gds.leiden.write(
    G,
    writeProperty="communities",
    includeIntermediateCommunities=True,
    relationshipWeightProperty="weight",
)

graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:__Community__) REQUIRE c.id IS UNIQUE;")

graph.query("""
MATCH (e:`__Entity__`)
UNWIND range(0, size(e.communities) - 1 , 1) AS index
CALL {
  WITH e, index
  WITH e, index
  WHERE index = 0
  MERGE (c:`__Community__` {id: toString(index) + '-' + toString(e.communities[index])})
  ON CREATE SET c.level = index
  MERGE (e)-[:IN_COMMUNITY]->(c)
  RETURN count(*) AS count_0
}
CALL {
  WITH e, index
  WITH e, index
  WHERE index > 0
  MERGE (current:`__Community__` {id: toString(index) + '-' + toString(e.communities[index])})
  ON CREATE SET current.level = index
  MERGE (previous:`__Community__` {id: toString(index - 1) + '-' + toString(e.communities[index - 1])})
  ON CREATE SET previous.level = index - 1
  MERGE (previous)-[:IN_COMMUNITY]->(current)
  RETURN count(*) AS count_1
}
RETURN count(*)
""")

# 处理完毕，删除内存中的子图投影
G.drop()

# 算法2：SLLPA(Speaker-Listener Label Propagation)算法
gds.graph.drop("communities", failIfMissing=False)
# 建立子图投影
G, result = gds.graph.project(
    "communities",  #  Graph name
    "__Entity__",  #  Node projection
    {
        "_ALL_": {
            "type": "*",
            "orientation": "UNDIRECTED",
            "properties": {"weight": {"property": "*", "aggregation": "COUNT"}},
        }
    },
)

# 找到调用sllpa算法的名字，gds.list()会返回GDS库中所有函数的Signature
algorithms = gds.list()
slpas =  algorithms[algorithms['description'].str.contains("Propagation", case=False, na=False)]
print(slpas)

# 调用sllpa算法	
gds.sllpa.write(
    G,
    maxIterations=100,
    minAssociationStrength = 0.1,
)

graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:__Community__) REQUIRE c.id IS UNIQUE;")

graph.query("""
MATCH (e:`__Entity__`)
UNWIND range(0, size(e.communityIds) - 1 , 1) AS index
CALL {
  WITH e, index
  MERGE (c:`__Community__` {id: '0-'+toString(e.communityIds[index])})
  ON CREATE SET c.level = 0
  MERGE (e)-[:IN_COMMUNITY]->(c)
  RETURN count(*) AS count_0
}
RETURN count(*)
""")

# 处理完毕，删除内存中的子图投影
G.drop()

# 社区摘要------------------------------------
# 按照SLLPA算法构建
# 为社区增加权重community_rank
graph.query("""
MATCH (c:`__Community__`)<-[:IN_COMMUNITY*]-(:`__Entity__`)<-[:MENTIONS]-(d:`__Chunk__`)
WITH c, count(distinct d) AS rank
SET c.community_rank = rank;
""")


# 检索社区所包含的结点与边的信息
community_info = graph.query("""
MATCH (c:`__Community__`)<-[:IN_COMMUNITY*]-(e:__Entity__)
WHERE c.level IN [0]
WITH c, collect(e ) AS nodes
WHERE size(nodes) > 1
CALL apoc.path.subgraphAll(nodes[0], {
	whitelistNodes:nodes
})
YIELD relationships
RETURN c.id AS communityId,
       [n in nodes | {id: n.id, description: n.description, type: [el in labels(n) WHERE el <> '__Entity__'][0]}] AS nodes,
       [r in relationships | {start: startNode(r).id, type: type(r), end: endNode(r).id, description: r.description}] AS rels
""")

community_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "给定一个输入三元组，生成信息摘要。没有序言。",
        ),
        ("human", community_template),
    ]
)

community_chain = community_prompt | llm | StrOutputParser()

# 转换社区信息为字符串
def prepare_string(data):
    nodes_str = "Nodes are:\n"
    for node in data['nodes']:
        node_id = node['id']
        node_type = node['type']
        if 'description' in node and node['description']:
            node_description = f", description: {node['description']}"
        else:
            node_description = ""
        nodes_str += f"id: {node_id}, type: {node_type}{node_description}\n"

    rels_str = "Relationships are:\n"
    for rel in data['rels']:
        start = rel['start']
        end = rel['end']
        rel_type = rel['type']
        if 'description' in rel and rel['description']:
            description = f", description: {rel['description']}"
        else:
            description = ""
        rels_str += f"({start})-[:{rel_type}]->({end}){description}\n"

    return nodes_str + "\n" + rels_str

# 封装社区摘要Chain为函数
def process_community(community):
    stringify_info = prepare_string(community)
    summary = community_chain.invoke({'community_info': stringify_info})
    return {"community": community['communityId'], "summary": summary, "full_content":stringify_info}

# 执行社区摘要  
summaries = []
for info in community_info:
    result = process_community(info)
    summaries.append(result)
    
# Store summaries
graph.query("""
UNWIND $data AS row
MERGE (c:__Community__ {id:row.community})
SET c.summary = row.summary, c.full_content = row.full_content
""", params={"data": summaries})
