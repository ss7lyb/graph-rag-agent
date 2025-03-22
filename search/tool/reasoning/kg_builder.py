import networkx as nx
from typing import Dict, List
import re
import time

class DynamicKnowledgeGraphBuilder:
    """
    动态知识图谱构建器
    
    在推理过程中实时构建与问题相关的知识子图，
    支持因果推理和关系发现
    """
    
    def __init__(self, graph, entity_relation_extractor=None):
        """
        初始化动态知识图谱构建器
        
        Args:
            graph: Neo4j图数据库连接
            entity_relation_extractor: 实体关系提取器
        """
        self.graph = graph
        self.extractor = entity_relation_extractor
        self.knowledge_graph = nx.DiGraph()  # 内存中的知识图谱
        self.seed_entities = set()  # 种子实体
        
    def build_query_graph(self, 
                        query: str, 
                        entities: List[str], 
                        depth: int = 2) -> nx.DiGraph:
        """
        为查询构建动态知识图谱
        
        Args:
            query: 用户查询
            entities: 初始实体列表
            depth: 图谱探索深度
            
        Returns:
            nx.DiGraph: 构建的知识图谱
        """
        # 重置图谱
        self.knowledge_graph = nx.DiGraph()
        self.seed_entities = set(entities)
        
        start_time = time.time()
        
        # 添加种子实体
        for entity in entities:
            self.knowledge_graph.add_node(
                entity, 
                type="seed_entity",
                properties={"source": "query"}
            )
        
        # 递归探索图谱
        self._explore_graph(entities, current_depth=0, max_depth=depth)
        
        # 添加图谱构建元数据
        self.knowledge_graph.graph['build_time'] = time.time() - start_time
        self.knowledge_graph.graph['query'] = query
        self.knowledge_graph.graph['entity_count'] = self.knowledge_graph.number_of_nodes()
        self.knowledge_graph.graph['relation_count'] = self.knowledge_graph.number_of_edges()
        
        print(f"构建查询图谱完成，包含 {self.knowledge_graph.number_of_nodes()} 个实体和 "
              f"{self.knowledge_graph.number_of_edges()} 个关系，耗时 "
              f"{time.time() - start_time:.2f}秒")
              
        return self.knowledge_graph
    
    def _explore_graph(self, entities: List[str], current_depth: int, max_depth: int):
        """
        递归探索和扩展图谱
        
        Args:
            entities: 当前层次的实体列表
            current_depth: 当前探索深度
            max_depth: 最大探索深度
        """
        if current_depth >= max_depth or not entities:
            return
            
        # 查询实体的相邻节点和关系
        try:
            # 构建查询
            query = """
            MATCH (e1:__Entity__)-[r]->(e2:__Entity__)
            WHERE e1.id IN $entity_ids
            RETURN e1.id AS source, 
                   e2.id AS target,
                   type(r) AS relation,
                   e2.description AS target_description
            LIMIT 100
            """
            
            # 执行查询
            relationships = self.graph.query(
                query, 
                params={"entity_ids": entities}
            )
            
            # 如果没有找到关系，返回
            if not relationships:
                return
                
            # 收集新发现的实体
            new_entities = []
            
            # 添加关系到图谱
            for rel in relationships:
                source = rel['source']
                target = rel['target']
                relation = rel['relation']
                
                # 检查目标实体是否已在图谱中
                if target not in self.knowledge_graph:
                    self.knowledge_graph.add_node(
                        target,
                        type="entity",
                        properties={"description": rel.get('target_description', '')}
                    )
                    new_entities.append(target)
                    
                # 添加边
                if not self.knowledge_graph.has_edge(source, target):
                    self.knowledge_graph.add_edge(
                        source, 
                        target, 
                        type=relation
                    )
            
            # 递归探索新发现的实体
            if new_entities:
                self._explore_graph(
                    new_entities, 
                    current_depth + 1, 
                    max_depth
                )
                
        except Exception as e:
            print(f"探索图谱时出错: {e}")
    
    def get_entity_neighborhood(self, entity: str, depth: int = 1) -> Dict:
        """
        获取实体的邻域信息
        
        Args:
            entity: 目标实体
            depth: 探索深度
            
        Returns:
            Dict: 包含邻域信息的字典
        """
        if entity not in self.knowledge_graph:
            return {"entity": entity, "neighbors": []}
        
        neighborhood = {"entity": entity, "neighbors": []}
        
        # BFS探索邻域
        visited = {entity}
        current_level = {entity}
        
        for _ in range(depth):
            next_level = set()
            
            # 遍历当前层的所有节点
            for node in current_level:
                # 获取所有出边邻居
                for successor in self.knowledge_graph.successors(node):
                    if successor not in visited:
                        edge_data = self.knowledge_graph.get_edge_data(node, successor)
                        neighborhood["neighbors"].append({
                            "from": node,
                            "to": successor,
                            "relation": edge_data.get("type", "unknown"),
                            "direction": "outgoing"
                        })
                        next_level.add(successor)
                        visited.add(successor)
                
                # 获取所有入边邻居
                for predecessor in self.knowledge_graph.predecessors(node):
                    if predecessor not in visited:
                        edge_data = self.knowledge_graph.get_edge_data(predecessor, node)
                        neighborhood["neighbors"].append({
                            "from": predecessor,
                            "to": node,
                            "relation": edge_data.get("type", "unknown"),
                            "direction": "incoming"
                        })
                        next_level.add(predecessor)
                        visited.add(predecessor)
            
            # 更新当前层
            current_level = next_level
            if not current_level:
                break
        
        return neighborhood
    
    def find_paths(self, source: str, target: str, max_paths: int = 3) -> List[List[Dict]]:
        """
        查找从源实体到目标实体的路径
        
        Args:
            source: 源实体
            target: 目标实体
            max_paths: 最大路径数
            
        Returns:
            List[List[Dict]]: 路径列表，每个路径是节点和关系的序列
        """
        # 检查实体是否在图谱中
        if source not in self.knowledge_graph or target not in self.knowledge_graph:
            return []
        
        try:
            # 使用NetworkX查找所有简单路径
            all_paths = list(nx.all_simple_paths(
                self.knowledge_graph, 
                source=source, 
                target=target, 
                cutoff=6  # 限制路径长度
            ))
            
            # 限制路径数量
            paths = all_paths[:max_paths]
            
            # 格式化路径
            formatted_paths = []
            for path in paths:
                formatted_path = []
                
                # 添加第一个节点
                formatted_path.append({
                    "type": "entity",
                    "id": path[0],
                    "properties": self.knowledge_graph.nodes[path[0]].get("properties", {})
                })
                
                # 添加后续节点和关系
                for i in range(len(path) - 1):
                    source_node = path[i]
                    target_node = path[i + 1]
                    
                    # 获取关系数据
                    edge_data = self.knowledge_graph.get_edge_data(source_node, target_node)
                    relation_type = edge_data.get("type", "unknown")
                    
                    # 添加关系
                    formatted_path.append({
                        "type": "relation",
                        "relation": relation_type,
                        "from": source_node,
                        "to": target_node
                    })
                    
                    # 添加目标节点
                    formatted_path.append({
                        "type": "entity",
                        "id": target_node,
                        "properties": self.knowledge_graph.nodes[target_node].get("properties", {})
                    })
                
                formatted_paths.append(formatted_path)
            
            return formatted_paths
            
        except Exception as e:
            print(f"查找路径时出错: {e}")
            return []
    
    def extract_subgraph_from_chunk(self, chunk_text: str, chunk_id: str) -> bool:
        """
        从文本块中提取知识子图
        
        Args:
            chunk_text: 文本块内容
            chunk_id: 文本块ID
            
        Returns:
            bool: 是否成功提取
        """
        if not self.extractor:
            return False
            
        try:
            # 使用实体关系提取器分析文本
            extraction_result = self.extractor._process_single_chunk(chunk_text)
            
            if not extraction_result:
                return False
                
            # 解析结果
            entity_pattern = re.compile(r'\("entity" : "(.+?)" : "(.+?)" : "(.+?)"\)')
            relationship_pattern = re.compile(r'\("relationship" : "(.+?)" : "(.+?)" : "(.+?)" : "(.+?)" : (.+?)\)')
            
            # 提取实体
            for match in entity_pattern.findall(extraction_result):
                entity_id, entity_type, description = match
                
                # 添加到图谱
                if entity_id not in self.knowledge_graph:
                    self.knowledge_graph.add_node(
                        entity_id,
                        type=entity_type,
                        properties={
                            "description": description,
                            "source": f"chunk:{chunk_id}"
                        }
                    )
            
            # 提取关系
            for match in relationship_pattern.findall(extraction_result):
                source_id, target_id, rel_type, description, weight = match
                
                # 确保节点存在
                for node_id in [source_id, target_id]:
                    if node_id not in self.knowledge_graph:
                        self.knowledge_graph.add_node(
                            node_id,
                            type="unknown",
                            properties={
                                "description": "从关系中提取的实体",
                                "source": f"chunk:{chunk_id}"
                            }
                        )
                
                # 添加关系
                self.knowledge_graph.add_edge(
                    source_id,
                    target_id,
                    type=rel_type,
                    properties={
                        "description": description,
                        "weight": float(weight),
                        "source": f"chunk:{chunk_id}"
                    }
                )
            
            return True
            
        except Exception as e:
            print(f"从文本块提取子图时出错: {e}")
            return False
    
    def get_central_entities(self, limit: int = 5) -> List[Dict]:
        """
        获取图谱中最重要的实体
        
        Args:
            limit: 返回实体数量
            
        Returns:
            List[Dict]: 重要实体列表
        """
        if not self.knowledge_graph.nodes:
            return []
            
        try:
            # 使用PageRank算法找出重要节点
            pagerank = nx.pagerank(self.knowledge_graph)
            
            # 排序
            top_entities = sorted(
                pagerank.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:limit]
            
            # 格式化结果
            result = []
            for entity_id, score in top_entities:
                node_data = self.knowledge_graph.nodes[entity_id]
                result.append({
                    "id": entity_id,
                    "centrality": score,
                    "type": node_data.get("type", "unknown"),
                    "properties": node_data.get("properties", {})
                })
                
            return result
            
        except Exception as e:
            print(f"计算中心实体时出错: {e}")
            # 使用度中心性作为备选方案
            in_degree = dict(self.knowledge_graph.in_degree())
            out_degree = dict(self.knowledge_graph.out_degree())
            
            # 合并入度和出度
            total_degree = {
                node: in_degree.get(node, 0) + out_degree.get(node, 0)
                for node in set(in_degree) | set(out_degree)
            }
            
            # 排序
            top_entities = sorted(
                total_degree.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:limit]
            
            # 格式化结果
            result = []
            for entity_id, degree in top_entities:
                node_data = self.knowledge_graph.nodes[entity_id]
                result.append({
                    "id": entity_id,
                    "degree": degree,
                    "type": node_data.get("type", "unknown"),
                    "properties": node_data.get("properties", {})
                })
                
            return result