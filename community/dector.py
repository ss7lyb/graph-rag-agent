from abc import ABC, abstractmethod
from graphdatascience import GraphDataScience
from langchain_community.graphs import Neo4jGraph
from typing import Tuple, Any, Dict
from contextlib import contextmanager

class BaseCommunityDetector(ABC):
    """
    社区检测的基类，定义了社区检测器的基本接口和共同功能。
    提供了完整的社区检测处理流程，包括图投影创建、社区检测、结果保存和资源清理。
    """
    def __init__(self, gds: GraphDataScience, graph: Neo4jGraph):
        self.gds = gds
        self.graph = graph
        self.projection_name = "communities"
        self.G = None

    @contextmanager
    def _graph_projection_context(self):
        """
        创建一个上下文管理器来处理图投影的生命周期，
        确保在使用完毕后正确清理资源，即使发生异常也能清理。
        """
        try:
            self.create_projection()
            yield
        finally:
            self.cleanup()
    
    def create_projection(self) -> Tuple[Any, Dict]:
        """创建图投影，返回投影图对象和结果信息。"""
        self.gds.graph.drop(self.projection_name, failIfMissing=False)
        
        self.G, result = self.gds.graph.project(
            self.projection_name,
            "__Entity__",
            {
                "_ALL_": {
                    "type": "*",
                    "orientation": "UNDIRECTED",
                    "properties": {"weight": {"property": "*", "aggregation": "COUNT"}},
                }
            },
        )
        return self.G, result
    
    def cleanup(self):
        """清理投影图资源"""
        if self.G:
            self.G.drop()
            self.G = None
            
    @abstractmethod
    def detect_communities(self):
        """社区检测的抽象方法，需要由子类实现"""
        pass
    
    @abstractmethod
    def save_communities(self):
        """保存社区检测结果的抽象方法，需要由子类实现"""
        pass

    def process(self) -> Dict[str, Any]:
        """
        执行完整的社区检测流程，包括：
        1. 创建图投影
        2. 检测社区
        3. 保存结果
        4. 清理资源
        
        Returns:
            Dict[str, Any]: 处理结果的统计信息
        """
        results = {
            'status': 'success',
            'algorithm': self.__class__.__name__,
            'details': {}
        }
        
        try:
            # 使用上下文管理器来处理图投影的生命周期
            with self._graph_projection_context():
                # 执行社区检测
                detection_result = self.detect_communities()
                if detection_result:
                    results['details']['detection'] = detection_result
                
                # 保存社区结果
                save_result = self.save_communities()
                if save_result:
                    results['details']['save'] = save_result
                
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
            raise
            
        return results

class LeidenDetector(BaseCommunityDetector):
    """使用Leiden算法进行社区检测的实现"""
    
    def detect_communities(self) -> Dict[str, Any]:
        """执行Leiden算法进行社区检测"""
        if not self.G:
            raise ValueError("请先创建图投影")
            
        # 检查连通分量信息
        wcc = self.gds.wcc.stats(self.G)
        
        # 执行Leiden算法
        result = self.gds.leiden.write(
            self.G,
            writeProperty="communities",
            includeIntermediateCommunities=True,
            relationshipWeightProperty="weight",
        )
        
        return {
            'componentCount': wcc['componentCount'],
            'componentDistribution': wcc['componentDistribution'],
            'communityCount': result.get('communityCount', 0)
        }
    
    def save_communities(self) -> Dict[str, int]:
        """保存Leiden算法的社区检测结果"""
        # 创建唯一性约束
        self.graph.query(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:__Community__) REQUIRE c.id IS UNIQUE;"
        )
        
        # 保存社区结构并返回统计信息
        result = self.graph.query("""
        MATCH (e:`__Entity__`)
        UNWIND range(0, size(e.communities) - 1 , 1) AS index
        CALL {
          WITH e, index
          WHERE index = 0
          MERGE (c:`__Community__` {id: toString(index) + '-' + toString(e.communities[index])})
          ON CREATE SET c.level = index
          MERGE (e)-[:IN_COMMUNITY]->(c)
          RETURN count(*) AS count_0
        }
        CALL {
          WITH e, index
          WHERE index > 0
          MERGE (current:`__Community__` {id: toString(index) + '-' + toString(e.communities[index])})
          ON CREATE SET current.level = index
          MERGE (previous:`__Community__` {id: toString(index - 1) + '-' + toString(e.communities[index - 1])})
          ON CREATE SET previous.level = index - 1
          MERGE (previous)-[:IN_COMMUNITY]->(current)
          RETURN count(*) AS count_1
        }
        RETURN count(*) as total_count
        """)
        
        return {'saved_communities': result[0]['total_count']}

class SLLPADetector(BaseCommunityDetector):
    """使用SLLPA算法进行社区检测的实现"""
    
    def detect_communities(self) -> Dict[str, Any]:
        """执行SLLPA算法进行社区检测"""
        if not self.G:
            raise ValueError("请先创建图投影")
            
        # 执行SLLPA算法
        result = self.gds.sllpa.write(
            self.G,
            maxIterations=100,
            minAssociationStrength=0.1,
        )
        
        return {
            'communityCount': result.get('communityCount', 0),
            'iterations': result.get('iterations', 0)
        }
    
    def save_communities(self) -> Dict[str, int]:
        """保存SLLPA算法的社区检测结果"""
        # 创建唯一性约束
        self.graph.query(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:__Community__) REQUIRE c.id IS UNIQUE;"
        )
        
        # 保存社区结构并返回统计信息
        result = self.graph.query("""
        MATCH (e:`__Entity__`)
        UNWIND range(0, size(e.communityIds) - 1 , 1) AS index
        CALL {
          WITH e, index
          MERGE (c:`__Community__` {id: '0-'+toString(e.communityIds[index])})
          ON CREATE SET c.level = 0
          MERGE (e)-[:IN_COMMUNITY]->(c)
          RETURN count(*) AS count_0
        }
        RETURN count(*) as total_count
        """)
        
        return {'saved_communities': result[0]['total_count']}