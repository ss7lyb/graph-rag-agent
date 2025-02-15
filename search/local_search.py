import os
from typing import Dict, Any
import pandas as pd
from neo4j import GraphDatabase, Result
from langchain_community.vectorstores import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.prompt import LC_SYSTEM_PROMPT
from dotenv import load_dotenv

class LocalSearch:
    """
    本地搜索类：使用Neo4j和LangChain实现基于向量检索的本地搜索功能。
    
    该类通过向量相似度搜索在知识图谱中查找相关内容，并生成回答。
    主要功能包括：
    1. 基于向量相似度的文本检索
    2. 社区内容和关系的检索
    3. 使用LLM生成最终答案
    """
    
    def __init__(self, llm, embeddings, response_type: str = "多个段落"):
        """
        初始化本地搜索器
        
        参数:
            llm: 大语言模型实例
            embeddings: 嵌入模型实例
            response_type: 响应类型，默认为"多个段落"
        """
        # 加载环境变量
        load_dotenv('../.env')
        
        # 保存模型实例和配置
        self.llm = llm
        self.embeddings = embeddings
        self.response_type = response_type
        
        # Neo4j数据库配置
        self.neo4j_uri = os.getenv('NEO4J_URI')
        self.neo4j_username = os.getenv('NEO4J_USERNAME')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD')
        
        # 初始化Neo4j驱动
        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_username, self.neo4j_password)
        )
        
        # 设置检索参数
        self.top_chunks = 3
        self.top_communities = 3
        self.top_outside_rels = 10
        self.top_inside_rels = 10
        self.top_entities = 10
        self.index_name = 'vector'
        
        # 初始化社区节点权重
        self._init_community_weights()
        
    def _init_community_weights(self):
        """初始化Neo4j中社区节点的权重"""
        self.db_query("""
        MATCH (n:`__Community__`)<-[:IN_COMMUNITY]-()<-[:MENTIONS]-(c)
        WITH n, count(distinct c) AS chunkCount
        SET n.weight = chunkCount
        """)
        
    def db_query(self, cypher: str, params: Dict[str, Any] = {}) -> pd.DataFrame:
        """执行Cypher查询并返回结果"""
        return self.driver.execute_query(
            cypher,
            parameters_=params,
            result_transformer_=Result.to_df
        )
        
    @property
    def retrieval_query(self) -> str:
        """获取Neo4j检索查询语句"""
        return """
        WITH collect(node) as nodes
        WITH
        collect {
            UNWIND nodes as n
            MATCH (n)<-[:MENTIONS]-(c:__Chunk__)
            WITH distinct c, count(distinct n) as freq
            RETURN {id:c.id, text: c.text} AS chunkText
            ORDER BY freq DESC
            LIMIT $topChunks
        } AS text_mapping,
        collect {
            UNWIND nodes as n
            MATCH (n)-[:IN_COMMUNITY]->(c:__Community__)
            WITH distinct c, c.community_rank as rank, c.weight AS weight
            RETURN c.summary 
            ORDER BY rank, weight DESC
            LIMIT $topCommunities
        } AS report_mapping,
        collect {
            UNWIND nodes as n
            MATCH (n)-[r]-(m:__Entity__) 
            WHERE NOT m IN nodes
            RETURN r.description AS descriptionText
            ORDER BY r.weight DESC 
            LIMIT $topOutsideRels
        } as outsideRels,
        collect {
            UNWIND nodes as n
            MATCH (n)-[r]-(m:__Entity__) 
            WHERE m IN nodes
            RETURN r.description AS descriptionText
            ORDER BY r.weight DESC 
            LIMIT $topInsideRels
        } as insideRels,
        collect {
            UNWIND nodes as n
            RETURN n.description AS descriptionText
        } as entities
        RETURN {
            Chunks: text_mapping, 
            Reports: report_mapping, 
            Relationships: outsideRels + insideRels, 
            Entities: entities
        } AS text, 1.0 AS score, {} AS metadata
        """
    
    def search(self, query: str) -> str:
        """
        执行本地搜索
        
        参数:
            query: 搜索查询字符串
            
        返回:
            str: 生成的最终答案
        """
        # 初始化对话提示模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", LC_SYSTEM_PROMPT),
            ("human", """
            ---分析报告--- 
            请注意，下面提供的分析报告按**重要性降序排列**。
            
            {report_data}
            
            用户的问题是：
            {question}
            """)
        ])
        
        # 创建搜索链
        chain = prompt | self.llm | StrOutputParser()
        
        # 初始化向量存储
        vector_store = Neo4jVector.from_existing_index(
            self.embeddings,
            url=self.neo4j_uri,
            username=self.neo4j_username,
            password=self.neo4j_password,
            index_name=self.index_name,
            retrieval_query=self.retrieval_query
        )
        
        # 执行相似度搜索
        docs = vector_store.similarity_search(
            query,
            k=self.top_entities,
            params={
                "topChunks": self.top_chunks,
                "topCommunities": self.top_communities,
                "topOutsideRels": self.top_outside_rels,
                "topInsideRels": self.top_inside_rels,
            }
        )
        
        # 使用LLM生成响应
        response = chain.invoke({
            "report_data": docs[0].page_content,
            "question": query,
            "response_type": self.response_type
        })

        print(docs[0].page_content) # 检索的数据源
        
        return response
        
    def close(self):
        """关闭Neo4j驱动连接"""
        self.driver.close()
        
    def __enter__(self):
        """上下文管理器入口"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()