from abc import ABC, abstractmethod
from graphdatascience import GraphDataScience
from langchain_community.graphs import Neo4jGraph
from typing import Tuple, Any, Dict
from contextlib import contextmanager
import time
import os
import psutil

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
        
        # 性能监控统计
        self.projection_time = 0
        self.detection_time = 0
        self.save_time = 0
        
        # 系统资源信息
        self.memory_mb = psutil.virtual_memory().total / (1024 * 1024)
        self.cpu_count = os.cpu_count() or 4
        
        # 根据系统资源动态调整参数
        self._adjust_parameters()
    
    def _adjust_parameters(self):
        """根据系统资源动态调整参数"""
        # 默认参数
        self.max_concurrency = min(4, self.cpu_count)
        
        # 根据内存大小调整参数
        memory_gb = self.memory_mb / 1024
        if memory_gb > 32:
            # 大内存系统
            self.node_count_limit = 100000
            self.timeout_seconds = 600
        elif memory_gb > 16:
            # 中等内存系统
            self.node_count_limit = 50000
            self.timeout_seconds = 300
        else:
            # 小内存系统
            self.node_count_limit = 20000
            self.timeout_seconds = 180
        
        print(f"社区检测参数调整: CPU核心数={self.cpu_count}, 内存={memory_gb:.1f}GB, "
              f"并发度={self.max_concurrency}, 节点限制={self.node_count_limit}")

    @contextmanager
    def _graph_projection_context(self):
        """
        创建一个上下文管理器来处理图投影的生命周期，
        确保在使用完毕后正确清理资源，即使发生异常也能清理。
        """
        try:
            projection_start = time.time()
            self.create_projection()
            self.projection_time = time.time() - projection_start
            yield
        except Exception as e:
            print(f"图投影处理时出错: {e}")
            raise
        finally:
            cleanup_start = time.time()
            self.cleanup()
            cleanup_time = time.time() - cleanup_start
            print(f"图投影清理完成，耗时: {cleanup_time:.2f}秒")
    
    def create_projection(self) -> Tuple[Any, Dict]:
        """创建图投影，返回投影图对象和结果信息。"""
        print("开始创建社区检测的图投影...")
        
        # 首先检查节点数量，避免过大的投影
        node_count = self._get_node_count()
        if node_count > self.node_count_limit:
            print(f"警告: 节点数量({node_count})超过限制({self.node_count_limit})，"
                  f"将使用过滤后的投影")
            return self._create_filtered_projection(node_count)
        
        # 尝试删除已存在的投影
        try:
            self.gds.graph.drop(self.projection_name, failIfMissing=False)
        except Exception as e:
            print(f"删除旧投影时出错 (可忽略): {e}")
        
        # 创建标准投影
        try:
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
            print(f"图投影创建成功，包含 {result.get('nodeCount', 0)} 个节点和 "
                  f"{result.get('relationshipCount', 0)} 个关系")
            return self.G, result
        except Exception as e:
            print(f"标准投影创建失败: {e}")
            # 尝试使用保守配置
            return self._create_conservative_projection()
    
    def _get_node_count(self) -> int:
        """获取实体节点数量"""
        result = self.graph.query(
            """
            MATCH (e:__Entity__)
            RETURN count(e) AS count
            """
        )
        return result[0]["count"] if result else 0
    
    def _create_filtered_projection(self, total_node_count: int) -> Tuple[Any, Dict]:
        """创建经过过滤的投影，以处理大型图"""
        print("创建过滤后的社区投影...")
        
        # 计算要保留的社区节点比例
        retention_ratio = min(1.0, self.node_count_limit / max(1, total_node_count))
        
        # 首先找出最重要的节点 (具有较多关系的节点)
        result = self.graph.query(
            """
            MATCH (e:__Entity__)-[r]-()
            WITH e, count(r) AS rel_count
            ORDER BY rel_count DESC
            LIMIT toInteger($limit)
            RETURN collect(id(e)) AS important_nodes
            """,
            params={"limit": self.node_count_limit}
        )
        
        if not result or not result[0]["important_nodes"]:
            print("无法获取重要节点，尝试使用保守投影")
            return self._create_conservative_projection()
        
        important_nodes = result[0]["important_nodes"]
        
        # 使用重要节点创建过滤投影
        try:
            node_filter = f"id(node) IN {important_nodes}"
            config = {
                "nodeProjection": {
                    "__Entity__": {
                        "properties": ["*"],
                        "filter": node_filter
                    }
                },
                "relationshipProjection": {
                    "_ALL_": {
                        "type": "*",
                        "orientation": "UNDIRECTED",
                        "properties": {"weight": {"property": "*", "aggregation": "COUNT"}}
                    }
                }
            }
            
            self.G, result = self.gds.graph.project(
                self.projection_name,
                config
            )
            print(f"过滤投影创建成功，包含 {result.get('nodeCount', 0)} 个节点和 "
                  f"{result.get('relationshipCount', 0)} 个关系")
            return self.G, result
        except Exception as e:
            print(f"过滤投影创建失败: {e}")
            return self._create_conservative_projection()
    
    def _create_conservative_projection(self) -> Tuple[Any, Dict]:
        """创建保守配置的投影，用于回退"""
        print("尝试使用保守配置创建投影...")
        
        try:
            # 使用最小配置创建投影
            config = {
                "nodeProjection": "__Entity__",
                "relationshipProjection": "*"
            }
            
            self.G, result = self.gds.graph.project(
                self.projection_name,
                config
            )
            print(f"保守投影创建成功，包含 {result.get('nodeCount', 0)} 个节点和 "
                  f"{result.get('relationshipCount', 0)} 个关系")
            return self.G, result
        except Exception as e:
            print(f"保守投影创建也失败: {e}")
            # 最后尝试: 只使用最重要的少量节点
            try:
                print("尝试创建最小化投影...")
                result = self.graph.query(
                    """
                    MATCH (e:__Entity__)-[r]-()
                    WITH e, count(r) AS rel_count
                    ORDER BY rel_count DESC
                    LIMIT 1000
                    RETURN collect(id(e)) AS critical_nodes
                    """
                )
                
                if not result or not result[0]["critical_nodes"]:
                    raise ValueError("无法获取关键节点")
                
                critical_nodes = result[0]["critical_nodes"]
                node_filter = f"id(node) IN {critical_nodes}"
                
                minimal_config = {
                    "nodeProjection": {
                        "__Entity__": {
                            "filter": node_filter
                        }
                    },
                    "relationshipProjection": "*"
                }
                
                self.G, result = self.gds.graph.project(
                    self.projection_name,
                    minimal_config
                )
                print(f"最小化投影创建成功，包含 {result.get('nodeCount', 0)} 个节点")
                return self.G, result
            except Exception as e2:
                print(f"所有投影方法均失败: {e2}")
                raise ValueError("无法创建社区检测所需的图投影")
    
    def cleanup(self):
        """清理投影图资源"""
        if self.G:
            try:
                self.G.drop()
                self.G = None
                print("社区投影图已清理")
            except Exception as e:
                print(f"清理投影图时出错: {e}")
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
        start_time = time.time()
        print(f"开始执行{self.__class__.__name__}社区检测流程...")
        
        results = {
            'status': 'success',
            'algorithm': self.__class__.__name__,
            'details': {}
        }
        
        try:
            # 使用上下文管理器来处理图投影的生命周期
            with self._graph_projection_context():
                # 执行社区检测
                detection_start = time.time()
                detection_result = self.detect_communities()
                self.detection_time = time.time() - detection_start
                
                if detection_result:
                    results['details']['detection'] = detection_result
                    
                print(f"社区检测完成，耗时: {self.detection_time:.2f}秒, "
                      f"找到 {detection_result.get('communityCount', 0)} 个社区")
                
                # 保存社区结果
                save_start = time.time()
                save_result = self.save_communities()
                self.save_time = time.time() - save_start
                
                if save_result:
                    results['details']['save'] = save_result
                    
                print(f"社区结果保存完成，耗时: {self.save_time:.2f}秒, "
                      f"保存了 {save_result.get('saved_communities', 0)} 个社区关系")
                
            # 添加性能统计
            total_time = time.time() - start_time
            performance = {
                'totalTime': total_time,
                'projectionTime': self.projection_time,
                'detectionTime': self.detection_time,
                'saveTime': self.save_time
            }
            results['performance'] = performance
            
            # 输出性能摘要
            print(f"\n{self.__class__.__name__}社区检测流程完成，总耗时: {total_time:.2f}秒")
            print(f"  投影创建: {self.projection_time:.2f}秒 ({self.projection_time/total_time*100:.1f}%)")
            print(f"  社区检测: {self.detection_time:.2f}秒 ({self.detection_time/total_time*100:.1f}%)")
            print(f"  结果保存: {self.save_time:.2f}秒 ({self.save_time/total_time*100:.1f}%)")
            
            return results
            
        except Exception as e:
            end_time = time.time()
            results['status'] = 'error'
            results['error'] = str(e)
            results['elapsed'] = end_time - start_time
            print(f"社区检测过程中出错: {e}")
            raise

class LeidenDetector(BaseCommunityDetector):
    """使用Leiden算法进行社区检测的实现"""
    
    def detect_communities(self) -> Dict[str, Any]:
        """执行Leiden算法进行社区检测"""
        if not self.G:
            raise ValueError("请先创建图投影")
            
        print("开始执行Leiden社区检测算法...")
        
        try:
            # 检查连通分量信息
            wcc = self.gds.wcc.stats(self.G)
            print(f"图包含 {wcc.get('componentCount', 0)} 个连通分量")
            
            # 根据系统资源动态调整Leiden参数
            leiden_params = self._get_optimized_leiden_params()
            
            # 执行Leiden算法
            result = self.gds.leiden.write(
                self.G,
                writeProperty="communities",
                includeIntermediateCommunities=True,
                relationshipWeightProperty="weight",
                **leiden_params
            )
            
            community_count = result.get('communityCount', 0)
            modularity = result.get('modularity', 0)
            ran_levels = result.get('ranLevels', 0)
            
            print(f"Leiden算法执行完成，检测到 {community_count} 个社区, "
                  f"模块度: {modularity:.4f}, 运行层级: {ran_levels}")
            
            return {
                'componentCount': wcc.get('componentCount', 0),
                'componentDistribution': wcc.get('componentDistribution', {}),
                'communityCount': community_count,
                'modularity': modularity,
                'ranLevels': ran_levels
            }
        except Exception as e:
            print(f"Leiden算法执行失败: {e}")
            print("尝试使用备用参数...")
            
            # 尝试使用更保守的参数
            try:
                fallback_result = self.gds.leiden.write(
                    self.G,
                    writeProperty="communities",
                    includeIntermediateCommunities=False,  # 禁用中间社区以节省内存
                    gamma=0.5,  # 更高的gamma值可能导致更大的社区
                    tolerance=0.001,  # 提高容忍度
                    maxLevels=2,  # 减少层级数
                    concurrency=1  # 单线程执行
                )
                
                print(f"备用Leiden算法执行完成，检测到 {fallback_result.get('communityCount', 0)} 个社区")
                
                return {
                    'communityCount': fallback_result.get('communityCount', 0),
                    'modularity': fallback_result.get('modularity', 0),
                    'ranLevels': fallback_result.get('ranLevels', 0),
                    'note': '使用了备用参数'
                }
            except Exception as e2:
                print(f"备用Leiden算法也失败: {e2}")
                raise ValueError(f"无法执行Leiden社区检测: {e}, 备用方法: {e2}")
    
    def _get_optimized_leiden_params(self) -> Dict[str, Any]:
        """根据系统资源返回优化的Leiden算法参数"""
        if self.memory_mb > 32 * 1024:  # 大于32GB内存
            return {
                'gamma': 1.0,
                'tolerance': 0.0001,
                'maxLevels': 10,
                'concurrency': self.max_concurrency
            }
        elif self.memory_mb > 16 * 1024:  # 大于16GB内存
            return {
                'gamma': 1.0,
                'tolerance': 0.0005,
                'maxLevels': 5,
                'concurrency': max(1, self.max_concurrency - 1)
            }
        else:  # 小内存系统
            return {
                'gamma': 0.8,
                'tolerance': 0.001,
                'maxLevels': 3,
                'concurrency': max(1, self.max_concurrency // 2)
            }
    
    def save_communities(self) -> Dict[str, int]:
        """保存Leiden算法的社区检测结果"""
        print("开始保存Leiden社区检测结果...")
        
        try:
            # 创建唯一性约束
            self.graph.query(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:__Community__) REQUIRE c.id IS UNIQUE;"
            )
            
            # 修复语法错误：使用正确的UNWIND+MATCH+MERGE模式
            base_result = self.graph.query("""
            MATCH (e:`__Entity__`)
            WHERE e.communities IS NOT NULL AND size(e.communities) > 0
            WITH collect({entityId: id(e), entity: e, community: e.communities[0]}) AS data
            UNWIND data AS item
            MERGE (c:`__Community__` {id: '0-' + toString(item.community)})
            ON CREATE SET c.level = 0
            WITH item, c
            MATCH (e) WHERE id(e) = item.entityId
            MERGE (e)-[:IN_COMMUNITY]->(c)
            RETURN count(*) AS base_count
            """)
            
            base_count = base_result[0]['base_count'] if base_result else 0
            print(f"已创建 {base_count} 个基础(0级)社区关系")
            
            # 使用类似的方法修复更高层级社区的处理
            higher_result = self.graph.query("""
            MATCH (e:`__Entity__`)
            WHERE e.communities IS NOT NULL AND size(e.communities) > 1
            WITH e, e.communities AS communities
            UNWIND range(1, size(communities) - 1) AS index
            WITH e, index, communities[index] AS current_community, communities[index-1] AS previous_community
            
            MERGE (current:`__Community__` {id: toString(index) + '-' + toString(current_community)})
            ON CREATE SET current.level = index
            WITH e, current, previous_community, index
            
            MATCH (previous:`__Community__` {id: toString(index - 1) + '-' + toString(previous_community)})
            MERGE (previous)-[:IN_COMMUNITY]->(current)
            
            RETURN count(*) AS higher_count
            """)
            
            higher_count = higher_result[0]['higher_count'] if higher_result else 0
            print(f"已创建 {higher_count} 个高级社区关系")
            
            total_count = base_count + higher_count
            
            return {'saved_communities': total_count}
        except Exception as e:
            print(f"保存Leiden社区结果时出错: {e}")
            # 尝试使用简化的保存方法
            try:
                print("尝试使用简化方法保存社区...")
                simplified_result = self.graph.query("""
                MATCH (e:`__Entity__`)
                WHERE e.communities IS NOT NULL AND size(e.communities) > 0
                // 只保存第0层社区
                MERGE (c:`__Community__` {id: '0-' + toString(e.communities[0])})
                ON CREATE SET c.level = 0
                MERGE (e)-[:IN_COMMUNITY]->(c)
                RETURN count(*) as count
                """)
                
                simplified_count = simplified_result[0]['count'] if simplified_result else 0
                print(f"使用简化方法保存了 {simplified_count} 个社区关系")
                
                return {'saved_communities': simplified_count, 'note': '使用了简化保存方法'}
            except Exception as e2:
                print(f"简化保存方法也失败: {e2}")
                raise ValueError(f"无法保存社区结果: {e}, 简化方法: {e2}")

class SLLPADetector(BaseCommunityDetector):
    """使用SLLPA算法进行社区检测的实现"""
    
    def detect_communities(self) -> Dict[str, Any]:
        """执行SLLPA算法进行社区检测"""
        if not self.G:
            raise ValueError("请先创建图投影")
            
        print("开始执行SLLPA社区检测算法...")
        
        try:
            # 根据系统资源动态调整SLLPA参数
            sllpa_params = self._get_optimized_sllpa_params()
            
            # 执行SLLPA算法
            result = self.gds.sllpa.write(
                self.G,
                writeProperty="communityIds",
                **sllpa_params
            )
            
            community_count = result.get('communityCount', 0)
            iterations = result.get('iterations', 0)
            
            print(f"SLLPA算法执行完成，检测到 {community_count} 个社区, "
                  f"迭代次数: {iterations}")
            
            return {
                'communityCount': community_count,
                'iterations': iterations
            }
        except Exception as e:
            print(f"SLLPA算法执行失败: {e}")
            print("尝试使用备用参数...")
            
            # 尝试使用更保守的参数
            try:
                fallback_result = self.gds.sllpa.write(
                    self.G,
                    writeProperty="communityIds",
                    maxIterations=50,  # 减少迭代次数
                    minAssociationStrength=0.2,  # 提高阈值
                    concurrency=1  # 单线程执行
                )
                
                print(f"备用SLLPA算法执行完成，检测到 {fallback_result.get('communityCount', 0)} 个社区")
                
                return {
                    'communityCount': fallback_result.get('communityCount', 0),
                    'iterations': fallback_result.get('iterations', 0),
                    'note': '使用了备用参数'
                }
            except Exception as e2:
                print(f"备用SLLPA算法也失败: {e2}")
                raise ValueError(f"无法执行SLLPA社区检测: {e}, 备用方法: {e2}")
    
    def _get_optimized_sllpa_params(self) -> Dict[str, Any]:
        """根据系统资源返回优化的SLLPA算法参数"""
        if self.memory_mb > 32 * 1024:  # 大于32GB内存
            return {
                'maxIterations': 100,
                'minAssociationStrength': 0.05,
                'concurrency': self.max_concurrency
            }
        elif self.memory_mb > 16 * 1024:  # 大于16GB内存
            return {
                'maxIterations': 80,
                'minAssociationStrength': 0.08,
                'concurrency': max(1, self.max_concurrency - 1)
            }
        else:  # 小内存系统
            return {
                'maxIterations': 50,
                'minAssociationStrength': 0.1,
                'concurrency': max(1, self.max_concurrency // 2)
            }
    
    def save_communities(self) -> Dict[str, int]:
        """保存SLLPA算法的社区检测结果"""
        print("开始保存SLLPA社区检测结果...")
        
        try:
            # 创建唯一性约束
            self.graph.query(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:__Community__) REQUIRE c.id IS UNIQUE;"
            )
            
            # 优化保存查询
            result = self.graph.query("""
            // 首先检查是否有实体具有communityIds属性
            MATCH (e:`__Entity__`)
            WHERE e.communityIds IS NOT NULL
            WITH count(e) AS entities_with_communities
            
            // 如果存在这样的实体，继续处理
            CALL {
                WITH entities_with_communities
                MATCH (e:`__Entity__`)
                WHERE e.communityIds IS NOT NULL
                // 使用批处理来优化性能
                WITH collect(e) AS entities
                CALL {
                    WITH entities
                    UNWIND entities AS e
                    UNWIND range(0, size(e.communityIds) - 1, 1) AS index
                    MERGE (c:`__Community__` {id: '0-'+toString(e.communityIds[index])})
                    ON CREATE SET c.level = 0, c.algorithm = 'SLLPA'
                    MERGE (e)-[:IN_COMMUNITY]->(c)
                }
                RETURN count(*) AS processed_count
            }
            
            RETURN CASE 
                WHEN entities_with_communities > 0 THEN entities_with_communities 
                ELSE 0 
            END AS total_count
            """)
            
            total_count = result[0]['total_count'] if result else 0
            print(f"已保存 {total_count} 个SLLPA社区关系")
            
            return {'saved_communities': total_count}
        except Exception as e:
            print(f"保存SLLPA社区结果时出错: {e}")
            # 尝试使用更简单的方法
            try:
                print("尝试使用简化方法保存SLLPA社区...")
                simplified_result = self.graph.query("""
                MATCH (e:`__Entity__`)
                WHERE e.communityIds IS NOT NULL AND size(e.communityIds) > 0
                WITH e, e.communityIds[0] AS primary_community
                MERGE (c:`__Community__` {id: '0-' + toString(primary_community)})
                ON CREATE SET c.level = 0, c.algorithm = 'SLLPA'
                MERGE (e)-[:IN_COMMUNITY]->(c)
                RETURN count(*) as count
                """)
                
                simplified_count = simplified_result[0]['count'] if simplified_result else 0
                print(f"使用简化方法保存了 {simplified_count} 个社区关系")
                
                return {'saved_communities': simplified_count, 'note': '使用了简化保存方法'}
            except Exception as e2:
                print(f"简化保存方法也失败: {e2}")
                raise ValueError(f"无法保存SLLPA社区结果: {e}, 简化方法: {e2}")