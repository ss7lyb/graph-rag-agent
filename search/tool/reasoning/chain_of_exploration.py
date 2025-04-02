from typing import List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ChainOfExplorationSearcher:
    """
    Chain of Exploration检索器
    
    一种能在图谱中自主探索并查找任务相关知识的方法
    """
    
    def __init__(self, graph, llm, embeddings_model):
        """
        初始化Chain of Exploration检索器
        
        Args:
            graph: 图数据库连接
            llm: 语言模型
            embeddings_model: 向量嵌入模型
        """
        self.graph = graph
        self.llm = llm
        self.embeddings = embeddings_model
        self.visited_nodes = set()
        self.exploration_path = []
        
    def explore(self, query: str, starting_entities: List[str], max_steps: int = 5):
        """
        从起始实体开始探索图谱
        
        Args:
            query: 用户查询
            starting_entities: 起始实体列表
            max_steps: 最大探索步数
            
        Returns:
            Dict: 探索结果
        """
        if not starting_entities:
            return {
                "entities": [],
                "relationships": [],
                "content": [],
                "exploration_path": []
            }
            
        self.visited_nodes = set(starting_entities)
        self.exploration_path = []
        query_embedding = self.embeddings.embed_query(query)
        
        # 添加起始节点到探索路径
        for entity in starting_entities:
            self.exploration_path.append({
                "step": 0,
                "node_id": entity,
                "action": "start",
                "reasoning": "初始实体"
            })
        
        current_entities = starting_entities
        results = {
            "entities": [],
            "relationships": [],
            "content": []
        }
        
        # 开始探索
        for step in range(max_steps):
            if not current_entities:
                break
                
            # 1. 找出邻居节点
            neighbors = self._get_neighbors(current_entities)
            if not neighbors:
                break
                
            # 2. 计算邻居节点的相关性
            scored_neighbors = self._score_neighbors(neighbors, query_embedding)
            
            # 3. 让LLM决定下一步探索方向
            next_entities, reasoning = self._decide_next_step(
                query, current_entities, scored_neighbors
            )
            
            # 4. 更新当前实体和已访问节点
            new_entities = [e for e in next_entities if e not in self.visited_nodes]
            self.visited_nodes.update(new_entities)
            current_entities = new_entities
            
            # 记录探索步骤
            for entity in new_entities:
                self.exploration_path.append({
                    "step": step + 1,
                    "node_id": entity,
                    "action": "explore",
                    "reasoning": reasoning
                })
                
            # 收集实体信息
            entity_info = self._get_entity_info(new_entities)
            results["entities"].extend(entity_info)
            
            # 收集关系信息
            rel_info = self._get_relationship_info(new_entities)
            results["relationships"].extend(rel_info)
            
            # 收集内容信息（如chunk）
            content_info = self._get_content_info(new_entities)
            results["content"].extend(content_info)
        
        # 添加探索路径
        results["exploration_path"] = self.exploration_path
        
        return results
            
    def _get_neighbors(self, entities):
        """获取实体的邻居节点"""
        try:
            query = """
            MATCH (e:__Entity__)-[r]-(neighbor:__Entity__)
            WHERE e.id IN $entity_ids AND NOT neighbor.id IN $visited_ids
            RETURN neighbor.id AS id, neighbor.description AS description,
                   type(r) AS relation_type, startNode(r).id AS source,
                   endNode(r).id AS target
            LIMIT 100
            """
            
            return self.graph.query(
                query, 
                params={
                    "entity_ids": entities, 
                    "visited_ids": list(self.visited_nodes)
                }
            )
        except Exception as e:
            print(f"获取邻居节点失败: {e}")
            return []
            
    def _score_neighbors(self, neighbors, query_embedding):
        """计算邻居节点与查询的相关性"""
        scored_neighbors = []
        
        for neighbor in neighbors:
            # 构建描述文本
            description = neighbor.get('description', '')
            
            try:
                # 计算相似度
                if description:
                    neighbor_embedding = self.embeddings.embed_query(description)
                    similarity = cosine_similarity(
                        np.array(query_embedding).reshape(1, -1),
                        np.array(neighbor_embedding).reshape(1, -1)
                    )[0][0]
                else:
                    similarity = 0.0
                    
                # 添加到评分列表
                scored_neighbors.append({
                    "id": neighbor['id'],
                    "description": description,
                    "relation_type": neighbor['relation_type'],
                    "source": neighbor['source'],
                    "target": neighbor['target'],
                    "similarity": similarity
                })
            except Exception as e:
                print(f"计算节点相似度失败: {e}")
                
        # 按相似度排序
        return sorted(scored_neighbors, key=lambda x: x['similarity'], reverse=True)
        
    def _decide_next_step(self, query, current_entities, scored_neighbors):
        """让LLM决定下一步探索方向"""
        # 构建提示
        prompt = f"""
        我正在使用Chain of Exploration方法探索知识图谱，以回答问题: "{query}"
        
        当前探索的实体有:
        {', '.join(current_entities)}
        
        下面是一些可能的下一步探索选项(按相关性排序):
        """
        
        # 添加前10个最相关的选项
        for i, neighbor in enumerate(scored_neighbors[:10]):
            prompt += f"{i+1}. {neighbor['id']} ({neighbor['similarity']:.2f}) - {neighbor['description']}\n"
            prompt += f"   通过关系 {neighbor['relation_type']} 连接到 {neighbor['source'] if neighbor['target'] in current_entities else neighbor['target']}\n\n"
            
        prompt += """
        请选择2-3个最有价值的实体，继续探索以回答问题。你的选择应该平衡相关性和覆盖广度。
        
        要求回复格式:
        ```
        selected: [实体1, 实体2, ...]
        reasoning: 你的选择理由...
        ```
        """
        
        try:
            # 调用LLM决策
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # 解析结果
            selected_line = None
            reasoning_line = None
            
            for line in content.split('\n'):
                if line.startswith('selected:'):
                    selected_line = line.replace('selected:', '').strip()
                elif line.startswith('reasoning:'):
                    reasoning_line = line.replace('reasoning:', '').strip()
            
            # 提取实体列表
            if selected_line:
                import re
                entities = re.findall(r'[\w\-]+', selected_line)
                return entities, reasoning_line or "未提供推理"
            else:
                # 如果无法解析，返回前3个最相关的
                return [n['id'] for n in scored_neighbors[:3]], "默认选择最相关的3个实体"
                
        except Exception as e:
            print(f"LLM决策失败: {e}")
            # 出错时使用简单启发式方法
            return [n['id'] for n in scored_neighbors[:2]], "出错时默认选择前2个实体"
    
    def _get_entity_info(self, entities):
        """获取实体详细信息"""
        if not entities:
            return []
            
        try:
            query = """
            MATCH (e:__Entity__)
            WHERE e.id IN $entity_ids
            RETURN e.id AS id, e.description AS description,
                   labels(e) AS labels
            """
            
            results = self.graph.query(query, params={"entity_ids": entities})
            return results if results else []
        except Exception as e:
            print(f"获取实体信息失败: {e}")
            return []
    
    def _get_relationship_info(self, entities):
        """获取实体关系信息"""
        if not entities:
            return []
            
        try:
            query = """
            MATCH (e1:__Entity__)-[r]-(e2:__Entity__)
            WHERE e1.id IN $entity_ids AND e2.id IN $visited_ids
            RETURN startNode(r).id AS source, endNode(r).id AS target,
                   type(r) AS type, r.description AS description,
                   r.weight AS weight
            """
            
            results = self.graph.query(
                query, 
                params={
                    "entity_ids": entities,
                    "visited_ids": list(self.visited_nodes)
                }
            )
            return results if results else []
        except Exception as e:
            print(f"获取关系信息失败: {e}")
            return []
    
    def _get_content_info(self, entities):
        """获取与实体相关的内容信息"""
        if not entities:
            return []
            
        try:
            query = """
            MATCH (c:__Chunk__)-[:MENTIONS]->(e:__Entity__)
            WHERE e.id IN $entity_ids
            RETURN DISTINCT c.id AS id, c.text AS text
            LIMIT 20
            """
            
            results = self.graph.query(query, params={"entity_ids": entities})
            return results if results else []
        except Exception as e:
            print(f"获取内容信息失败: {e}")
            return []