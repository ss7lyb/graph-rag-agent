import os
from graphdatascience import GraphDataScience
from typing import Tuple, List, Any
from dataclasses import dataclass
from langchain_community.graphs import Neo4jGraph

from config.settings import similarity_threshold

from dotenv import load_dotenv

load_dotenv('../.env')

@dataclass
class GDSConfig:
    """Neo4j GDS配置参数"""
    uri: str = os.environ["NEO4J_URI"]
    username: str = os.environ["NEO4J_USERNAME"]
    password: str = os.environ["NEO4J_PASSWORD"]
    similarity_threshold: float = similarity_threshold
    word_edit_distance: int = 3

class SimilarEntityDetector:
    """
    相似实体检测器，使用Neo4j GDS库实现实体相似性分析和社区识别。
    
    主要功能：
    1. 建立实体投影图
    2. 使用KNN算法识别相似实体
    3. 使用WCC算法进行社区检测
    4. 识别潜在的重复实体
    """
    
    def __init__(self, config: GDSConfig):
        """
        初始化相似实体检测器
        
        Args:
            config: GDS配置参数，包含连接信息和算法阈值
        """
        self.config = config
        self.gds = GraphDataScience(
            self.config.uri,
            auth=(self.config.username, self.config.password)
        )
        self.graph = Neo4jGraph()
        self.projection_name = "entities"
        self.G = None
        
    def create_entity_projection(self) -> Tuple[Any, Any]:
        """
        创建实体的内存投影子图
        
        Returns:
            Tuple[Any, Any]: 投影图对象和结果信息
        """
        # 如果已存在，先清除旧的投影
        self.gds.graph.drop(self.projection_name, failIfMissing=False)
        
        # 创建新的投影图
        self.G, result = self.gds.graph.project(
            self.projection_name,          # 图名称
            "__Entity__",                  # 节点投影
            "*",                           # 关系投影（所有类型）
            nodeProperties=["embedding"]    # 配置参数
        )
        
        return self.G, result
        
    def detect_similar_entities(self) -> None:
        """
        使用KNN算法检测相似实体并创建SIMILAR关系
        """
        if not self.G:
            raise ValueError("请先创建实体投影")
            
        # 使用KNN算法找出相似实体
        self.gds.knn.mutate(
            self.G,
            nodeProperties=['embedding'],
            mutateRelationshipType='SIMILAR',
            mutateProperty='score',
            similarityCutoff=self.config.similarity_threshold
        )
        
        # 将KNN结果写入数据库
        self.gds.knn.write(
            self.G,
            nodeProperties=['embedding'],
            writeRelationshipType='SIMILAR',
            writeProperty='score',
            similarityCutoff=self.config.similarity_threshold
        )
        
    def detect_communities(self) -> None:
        """
        使用WCC算法检测社区并将结果写入节点的wcc属性
        """
        if not self.G:
            raise ValueError("请先创建实体投影")
            
        self.gds.wcc.write(
            self.G,
            writeProperty="wcc",
            relationshipTypes=["SIMILAR"]
        )
        
    def find_potential_duplicates(self) -> List[Any]:
        """
        查找潜在的重复实体
        
        Returns:
            List[Any]: 潜在重复实体的候选列表
        """
        return self.graph.query(
            """
            MATCH (e:`__Entity__`)
            WHERE size(e.id) > 1  // 长度大于2个字符
            WITH e.wcc AS community, collect(e) AS nodes, count(*) AS count
            WHERE count > 1
            UNWIND nodes AS node
            // 添加文本距离计算
            WITH distinct
                [n IN nodes WHERE apoc.text.distance(toLower(node.id), toLower(n.id)) < $distance | n.id] 
                AS intermediate_results
            WHERE size(intermediate_results) > 1
            WITH collect(intermediate_results) AS results
            // 如果组之间有共同元素，则合并组
            UNWIND range(0, size(results)-1, 1) as index
            WITH results, index, results[index] as result
            WITH apoc.coll.sort(reduce(acc = result, 
                index2 IN range(0, size(results)-1, 1) |
                CASE WHEN index <> index2 AND
                    size(apoc.coll.intersection(acc, results[index2])) > 0
                    THEN apoc.coll.union(acc, results[index2])
                    ELSE acc
                END
            )) as combinedResult
            WITH distinct(combinedResult) as combinedResult
            // 额外过滤
            WITH collect(combinedResult) as allCombinedResults
            UNWIND range(0, size(allCombinedResults)-1, 1) as combinedResultIndex
            WITH allCombinedResults[combinedResultIndex] as combinedResult, 
                 combinedResultIndex, 
                 allCombinedResults
            WHERE NOT any(x IN range(0,size(allCombinedResults)-1,1)
                WHERE x <> combinedResultIndex
                AND apoc.coll.containsAll(allCombinedResults[x], combinedResult)
            )
            RETURN combinedResult
            """,
            params={'distance': self.config.word_edit_distance}
        )
        
    def cleanup(self) -> None:
        """清理内存中的投影图"""
        if self.G:
            self.G.drop()
            self.G = None

    def process_entities(self) -> List[Any]:
        """
        执行完整的实体处理流程
        
        Returns:
            List[Any]: 潜在重复实体的列表
        """
        try:
            # 创建实体投影
            self.create_entity_projection()
            
            # 检测相似实体
            self.detect_similar_entities()
            
            # 检测社区
            self.detect_communities()
            
            # 查找潜在重复
            duplicates = self.find_potential_duplicates()
            
            return duplicates
            
        finally:
            # 确保清理投影图
            self.cleanup()