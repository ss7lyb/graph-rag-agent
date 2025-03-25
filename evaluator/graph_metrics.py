from typing import Dict, List, Tuple
from evaluator.base import BaseMetric
import re

from evaluator.utils import normalize_answer

class CommunityRelevanceMetric(BaseMetric):
    """社区相关性评估指标"""
    
    metric_name = "community_relevance"
    
    def __init__(self, config):
        super().__init__(config)
        self.neo4j_client = config.get('neo4j_client', None)
    
    def calculate_metric(self, data) -> Tuple[Dict[str, float], List[float]]:
        """计算社区相关性"""
        relevance_scores = []
        
        for sample in data.samples:
            question = sample.question
            agent_type = sample.agent_type.lower() if sample.agent_type else ""
            
            # 提取问题关键词
            keywords = re.findall(r'\b[\w\u4e00-\u9fa5]{2,}\b', normalize_answer(question))
            keywords = [k for k in keywords if len(k) > 1 and len(k) < 15]
            
            # 特殊处理naive代理
            if agent_type == "naive":
                chunks = sample.referenced_entities  # 可能存放的是文本块ID
                
                # 查询文本块关联的社区
                community_info = ""
                if self.neo4j_client and chunks:
                    try:
                        # 先查询文本块关联的实体
                        query = """
                        MATCH (c:__Chunk__)-[:MENTIONS]->(e:__Entity__)
                        WHERE c.id IN $chunk_ids
                        RETURN COLLECT(DISTINCT e.id) AS entity_ids
                        """
                        result = self.neo4j_client.execute_query(query, {"chunk_ids": chunks})
                        
                        entity_ids = []
                        if result.records and result.records[0].get("entity_ids"):
                            entity_ids = result.records[0].get("entity_ids")
                        
                        # 查询与这些实体相关的社区
                        if entity_ids:
                            community_query = """
                            MATCH (c:__Community__)
                            WHERE ANY(entity_id IN c.communities WHERE entity_id IN $entity_ids)
                            RETURN c.summary AS summary, c.full_content AS full_content
                            LIMIT 3
                            """
                            community_result = self.neo4j_client.execute_query(
                                community_query, {"entity_ids": entity_ids}
                            )
                            
                            if community_result.records:
                                for record in community_result.records:
                                    summary = record.get("summary", "")
                                    full_content = record.get("full_content", "")
                                    if summary:
                                        community_info += summary + " "
                                    if full_content:
                                        community_info += full_content + " "
                    except Exception as e:
                        print(f"查询文本块关联社区时出错: {e}")
                
                # 计算基于社区内容的相关性得分
                if community_info and keywords:
                    matched = sum(1 for k in keywords if k.lower() in community_info.lower())
                    match_rate = matched / len(keywords) if keywords else 0
                    
                    # 基础分0.3，匹配率最多贡献0.4分
                    score = 0.3 + 0.4 * match_rate
                else:
                    # 没有社区信息，给予基础分
                    score = 0.3 + 0.1 * len(chunks) / 3  # 每个文本块增加一点分数
                    score = min(0.4, score)  # 最多0.4分
                
                relevance_scores.append(score)
                continue
            
            # 处理其他代理的社区相关性
            entity_ids = sample.referenced_entities
            
            # 查询与实体关联的社区
            community_info = ""
            if self.neo4j_client and entity_ids:
                try:
                    # 查询社区信息
                    query = """
                    MATCH (c:__Community__)
                    WHERE ANY(entity_id IN c.communities WHERE entity_id IN $entity_ids)
                    RETURN c.summary AS summary, c.full_content AS full_content
                    LIMIT 5
                    """
                    result = self.neo4j_client.execute_query(query, {"entity_ids": entity_ids})
                    
                    if result.records:
                        for record in result.records:
                            summary = record.get("summary", "")
                            full_content = record.get("full_content", "")
                            if summary:
                                community_info += summary + " "
                            if full_content:
                                community_info += full_content + " "
                except Exception as e:
                    print(f"查询社区信息失败: {e}")
            
            # 计算相关性得分
            if community_info and keywords:
                matched = sum(1 for k in keywords if k.lower() in community_info.lower())
                match_rate = matched / len(keywords) if keywords else 0
                
                # 基础分根据代理类型不同
                base_score = 0.3
                if agent_type == "graph":
                    base_score = 0.4
                    match_rate *= 1.2  # 给graph代理更高加成
                elif agent_type == "hybrid":
                    base_score = 0.35
                    match_rate *= 1.1  # 给hybrid代理小幅加成
                
                # 计算最终分数
                score = base_score + 0.5 * match_rate
                score = min(1.0, score)  # 确保不超过1.0
            else:
                # 没有社区信息，基于代理类型给予基础分
                if agent_type == "graph":
                    score = 0.4
                elif agent_type == "hybrid":
                    score = 0.35
                else:
                    score = 0.3
            
            relevance_scores.append(score)
        
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        
        return {"community_relevance": avg_relevance}, relevance_scores


class SubgraphQualityMetric(BaseMetric):
    """
    子图质量评估指标 - 评估检索到的子图的质量和信息密度
    """
    
    metric_name = "subgraph_quality"
    
    def __init__(self, config):
        super().__init__(config)
        self.density_weight = 0.5
        self.connectivity_weight = 0.5
    
    def calculate_metric(self, data) -> Tuple[Dict[str, float], List[float]]:
        """
        计算子图质量
        
        Args:
            data: 评估数据
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 总体得分和每个样本的得分
        """
        quality_scores = []
        
        for sample in data.samples:
            entities = sample.retrieved_entities
            relationships = sample.referenced_relationships
            
            # 如果没有实体或关系，质量为0
            if not entities or not relationships:
                quality_scores.append(0.0)
                continue
            
            # 计算图密度 - 边数与最大可能边数之比
            nodes_count = len(entities)
            edges_count = len(relationships)
            
            max_edges = nodes_count * (nodes_count - 1) / 2  # 最大可能的边数
            density = edges_count / max_edges if max_edges > 0 else 0
            
            # 计算连通性 - 检查有多少实体参与了关系
            entity_in_rel = set()
            for src, _, dst in relationships:
                entity_in_rel.add(src)
                entity_in_rel.add(dst)
            
            connectivity = len(entity_in_rel) / nodes_count if nodes_count > 0 else 0
            
            # 加权平均
            quality = density * self.density_weight + connectivity * self.connectivity_weight
            
            # 根据代理类型调整
            if sample.agent_type == "graph":
                quality = min(1.0, quality * 1.2)  # graph代理奖励
            
            quality_scores.append(quality)
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        return {"subgraph_quality": avg_quality}, quality_scores


class GraphCoverageMetric(BaseMetric):
    """图覆盖率评估指标"""
    
    metric_name = "graph_coverage"
    
    def __init__(self, config):
        super().__init__(config)
        self.neo4j_client = config.get('neo4j_client', None)
    
    def calculate_metric(self, data) -> Tuple[Dict[str, float], List[float]]:
        """计算图覆盖率"""
        coverage_scores = []
        
        for sample in data.samples:
            question = sample.question
            agent_type = sample.agent_type.lower() if sample.agent_type else ""
            
            # 提取关键词
            keywords = re.findall(r'\b[\w\u4e00-\u9fa5]{2,}\b', normalize_answer(question))
            keywords = [k for k in keywords if len(k) > 1 and len(k) < 15]
            
            # 特殊处理naive代理
            if agent_type == "naive":
                chunks = sample.referenced_entities  # 可能存放的是文本块ID
                chunk_count = len(chunks) if chunks else 0
                
                # 由于naive不使用图结构，根据文本块数量给基础分
                base_score = 0.3
                chunk_bonus = min(0.3, chunk_count * 0.1)  # 每个文本块加0.1，最多加0.3
                
                coverage_scores.append(base_score + chunk_bonus)
                continue
            
            # 处理其他代理
            entity_ids = sample.referenced_entities
            rel_ids = []
            if isinstance(sample.referenced_relationships, list):
                rel_ids = [r for r in sample.referenced_relationships if isinstance(r, str)]
            
            # 计算基于图结构的覆盖率
            entity_count = len(entity_ids) if entity_ids else 0
            rel_count = len(rel_ids) if rel_ids else 0
            
            # 查询实体信息以匹配关键词
            entities_text = ""
            if self.neo4j_client and entity_ids:
                try:
                    query = """
                    MATCH (e:__Entity__)
                    WHERE e.id IN $ids
                    RETURN e.id AS id, e.description AS description
                    """
                    result = self.neo4j_client.execute_query(query, {"ids": entity_ids})
                    
                    if result.records:
                        for record in result.records:
                            entity_id = record.get("id", "")
                            entity_desc = record.get("description", "")
                            if entity_id:
                                entities_text += f"{entity_id} {entity_desc} "
                except Exception as e:
                    print(f"查询实体信息失败: {e}")
            
            # 计算关键词匹配
            keyword_match = 0
            if keywords and entities_text:
                for keyword in keywords:
                    if keyword.lower() in entities_text.lower():
                        keyword_match += 1
                
            # 计算结构得分和关键词得分
            structure_score = min(0.5, 0.1 * entity_count + 0.05 * rel_count)
            keyword_score = 0.5 * (keyword_match / len(keywords)) if keywords else 0
            
            # 根据代理类型调整基础分
            base_score = 0.2
            if agent_type == "graph":
                base_score = 0.3
                structure_score *= 1.2  # 给graph代理结构分加成
            elif agent_type == "hybrid":
                base_score = 0.25
                structure_score *= 1.1  # 给hybrid代理小幅加成
            
            # 计算最终分数
            score = base_score + structure_score + keyword_score
            score = min(1.0, score)  # 确保不超过1.0
            
            coverage_scores.append(score)
        
        avg_coverage = sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0.0
        
        return {"graph_coverage": avg_coverage}, coverage_scores