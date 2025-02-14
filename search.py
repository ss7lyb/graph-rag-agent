import os

from dotenv import load_dotenv
from neo4j import GraphDatabase, Result
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm
from typing import Dict, Any
import pandas as pd

from model.get_models import get_llm_model, get_embeddings_model
from config.prompt import LC_SYSTEM_PROMPT, MAP_SYSTEM_PROMPT, REDUCE_SYSTEM_PROMPT

load_dotenv()

llm = get_llm_model()
embeddings = get_embeddings_model()

# LLM以多段的形式回答问题
response_type: str = "多个段落"

# 设置Neo4j的运行参数
NEO4J_URI=os.getenv('NEO4J_URI')
NEO4J_USERNAME=os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD=os.getenv('NEO4J_PASSWORD')

# Neo4j向量索引的名字
index_name = "vector"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# 执行Cypher查询
def db_query(cypher: str, params: Dict[str, Any] = {}) -> pd.DataFrame:
    """Executes a Cypher statement and returns a DataFrame"""
    return driver.execute_query(
        cypher, parameters_=params, result_transformer_=Result.to_df
    )

# 为社区结点设置权重以便检索时排序
db_query("""
MATCH (n:`__Community__`)<-[:IN_COMMUNITY]-()<-[:MENTIONS]-(c)
WITH n, count(distinct c) AS chunkCount
SET n.weight = chunkCount
""")

# 局部查询 local search ---------------------------------------------
topChunks = 3
topCommunities = 3
topOutsideRels = 10
topInsideRels = 10
topEntities = 10

lc_retrieval_query = """
WITH collect(node) as nodes
// Entity - Text Unit Mapping
WITH
collect {
    UNWIND nodes as n
    MATCH (n)<-[:MENTIONS]-(c:__Chunk__)
    WITH distinct c, count(distinct n) as freq
    RETURN {id:c.id, text: c.text} AS chunkText
    ORDER BY freq DESC
    LIMIT $topChunks
} AS text_mapping,
// Entity - Report Mapping
collect {
    UNWIND nodes as n
    MATCH (n)-[:IN_COMMUNITY]->(c:__Community__)
    WITH distinct c, c.community_rank as rank, c.weight AS weight
    RETURN c.summary 
    ORDER BY rank, weight DESC
    LIMIT $topCommunities
} AS report_mapping,
// Outside Relationships 
collect {
    UNWIND nodes as n
    MATCH (n)-[r]-(m:__Entity__) 
    WHERE NOT m IN nodes
    RETURN r.description AS descriptionText
    ORDER BY r.weight DESC 
    LIMIT $topOutsideRels
} as outsideRels,
// Inside Relationships 
collect {
    UNWIND nodes as n
    MATCH (n)-[r]-(m:__Entity__) 
    WHERE m IN nodes
    RETURN r.description AS descriptionText
    ORDER BY r.weight DESC 
    LIMIT $topInsideRels
} as insideRels,
// Entities description
collect {
    UNWIND nodes as n
    RETURN n.description AS descriptionText
} as entities
// We don't have covariates or claims here
RETURN {Chunks: text_mapping, Reports: report_mapping, 
       Relationships: outsideRels + insideRels, 
       Entities: entities} AS text, 1.0 AS score, {} AS metadata
"""

# 局部检索器
def local_retriever(query: str, response_type: str = response_type) -> str:
    # 局部检索的提示词
    lc_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                LC_SYSTEM_PROMPT,
            ),
            (
                "human",
                """
                ---分析报告--- 
                请注意，下面提供的分析报告按**重要性降序排列**。
                
                {report_data}
                

                用户的问题是：
                {question}
                """,
            ),
        ]
    )
    # 局部检索的chain
    lc_chain = lc_prompt | llm | StrOutputParser()
    # 局部检索的Neo4j向量存储与索引
    lc_vector = Neo4jVector.from_existing_index(
        embeddings,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name=index_name,
        retrieval_query=lc_retrieval_query,
    )
    # 先进行向量相似性搜索
    docs = lc_vector.similarity_search(
        query,
        k=topEntities,
        params={
            "topChunks": topChunks,
            "topCommunities": topCommunities,
            "topOutsideRels": topOutsideRels,
            "topInsideRels": topInsideRels,
        },
    )
    
    print(docs[0].page_content)
    
    # 向量相似性搜索的结果注入提示词并提交给LLM
    lc_response = lc_chain.invoke(
        {
            "report_data": docs[0].page_content,
            "question": query,
            "response_type": response_type,
        }
    )
    # 返回LLM的答复
    return lc_response

answer1 = local_retriever("去取经之前孙悟空被关在了什么地方？")

print(answer1)

# 全局查询 global search--------------------------------------
# 全局检索器
def global_retriever(query: str, level: int, response_type: str = response_type) -> str:
    # MAP阶段生成中间结果的prompt与chain
    map_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                MAP_SYSTEM_PROMPT,
            ),
            (
                "human",
                """
                ---数据表格--- 
                {context_data}
                
                
                用户的问题是：
                {question}
                """,
            ),
        ]
    )
    map_chain = map_prompt | llm | StrOutputParser()
    # Reduce阶段生成最终结果的prompt与chain
    reduce_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                REDUCE_SYSTEM_PROMPT,
            ),
            (
                "human",
                """
                ---分析报告--- 
                {report_data}


                用户的问题是：
                {question}
                """,
            ),
        ]
    )
    reduce_chain = reduce_prompt | llm | StrOutputParser()

    # 连接Neo4j
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        refresh_schema=False,
    )
    # 检索指定层级的社区
    community_data = graph.query(
        """
        MATCH (c:__Community__)
        WHERE c.level = 0
        RETURN {communityId:c.id, full_content:c.full_content} AS output
        """,
        params={"level": level},
    )
    # 用LLM从每个检索到的社区摘要生成中间结果
    intermediate_results = []
    for community in tqdm(community_data, desc="Processing communities"):
        intermediate_response = map_chain.invoke(
            {"question": query, "context_data": community["output"]}
        )
        intermediate_results.append(intermediate_response)
        # 输出看一下
        print(intermediate_response)
    # 再用LLM从每个社区摘要生成的中间结果生成最终的答复
    final_response = reduce_chain.invoke(
        {
            "report_data": intermediate_results,
            "question": query,
            "response_type": response_type,
        }
    )
    # 返回LLM最终的答复
    return final_response

answer2 = global_retriever("孙悟空对唐僧是一种什么样的心态？", 0)
print(answer2)