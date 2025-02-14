from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from typing import List
from model.get_models import get_embeddings_model, get_llm_model

from dotenv import load_dotenv

load_dotenv('../.env')

class EntityIndexManager:
    """
    实体索引管理器，负责在Neo4j数据库中创建和管理实体的向量索引。
    处理实体节点的embedding向量计算和索引创建，支持后续基于向量相似度的实体查询。
    """
    
    def __init__(self, refresh_schema: bool = True):
        """
        初始化实体索引管理器
        
        Args:
            refresh_schema: 是否刷新Neo4j图数据库的schema
        """
        # 初始化图数据库连接
        self.graph = Neo4jGraph(refresh_schema=refresh_schema)
        
        # 初始化模型
        self.embeddings = get_embeddings_model()
        self.llm = get_llm_model()
        
    def clear_existing_index(self) -> None:
        """清除已存在的实体embedding索引"""
        self.graph.query("DROP INDEX entity_embedding IF EXISTS")

    def create_entity_index(self, 
                          node_label: str = '__Entity__',
                          text_properties: List[str] = ['id', 'description'],
                          embedding_property: str = 'embedding') -> Neo4jVector:
        """
        创建实体的向量索引
        
        Args:
            node_label: 实体节点的标签
            text_properties: 用于计算embedding的文本属性列表
            embedding_property: 存储embedding的属性名
            
        Returns:
            Neo4jVector: 创建的向量存储对象
        """
        # 先清除已有索引
        self.clear_existing_index()
        
        # 创建新的向量索引
        vector_store = Neo4jVector.from_existing_graph(
            self.embeddings,
            node_label=node_label,
            text_node_properties=text_properties,
            embedding_node_property=embedding_property
        )
        
        return vector_store