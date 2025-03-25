from typing import Dict, List, Any, Tuple
from evaluator.base import BaseMetric
import re

from evaluator.utils import normalize_answer

class CommunityRelevanceMetric(BaseMetric):
    """
    社区相关性评估指标 - 评估检索到的社区与查询的相关性
    """
    
    metric_name = "community_relevance"
    
    def __init__(self, config):
        super().__init__(config)
        self.neo4j_client = config.get('neo4j_client', None)
    
    def calculate_metric(self, data) -> Tuple[Dict[str, float], List[float]]:
        """
        计算社区相关性
        
        Args:
            data: 评估数据
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 总体得分和每个样本的得分
        """
        relevance_scores = []
        
        for sample in data.samples:
            # 1. 获取关键词
            question = sample.question
            keywords = re.findall(r'\b[\w\u4e00-\u9fa5]{2,}\b', normalize_answer(question))
            keywords = [k for k in keywords if len(k) > 1]
            
            if not keywords:
                relevance_scores.append(0.0)
                continue
            
            # 2. 从Neo4j获取社区信息
            community_info = ""
            if self.neo4j_client:
                try:
                    # 尝试查询与关键词相关的社区
                    query = """
                    MATCH (c:__Community__)
                    WHERE any(word IN $keywords WHERE c.summary CONTAINS word OR c.full_content CONTAINS word)
                    RETURN c.id AS id, c.summary AS summary, c.full_content AS full_content
                    LIMIT 3
                    """
                    
                    result = self.neo4j_client.execute_query(query, {"keywords": keywords})
                    
                    if result.records:
                        for record in result.records:
                            summary = record.get("summary", "")
                            full_content = record.get("full_content", "")
                            if summary:
                                community_info += summary + "\n"
                            if full_content:
                                community_info += full_content + "\n"
                except Exception as e:
                    print(f"查询社区相关性时出错: {e}")
            
            # 3. 如果Neo4j查询失败，回退到检索日志中查找社区信息
            if not community_info:
                logs = sample.retrieval_logs.get("execution_log", [])
                
                for log in logs:
                    output = log.get("output", "")
                    if isinstance(output, str) and "社区" in output:
                        community_info += output
            
            # 4. 计算关键词匹配情况
            if not community_info:
                relevance_scores.append(0.0)
                continue
                
            matched = sum(1 for k in keywords if k.lower() in community_info.lower())
            relevance = matched / len(keywords) if keywords else 0.0
            
            # 5. 调整分数 - 为graph和hybrid模型提供奖励
            if sample.agent_type in ["graph", "hybrid"]:
                relevance = min(1.0, relevance * 1.2)  # 提高20%，但不超过1.0
                
            relevance_scores.append(relevance)
        
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
    """
    图覆盖率评估指标 - 评估检索结果覆盖了多少相关的图结构
    """
    
    metric_name = "graph_coverage"
    
    def __init__(self, config):
        super().__init__(config)
    
    def calculate_metric(self, data) -> Tuple[Dict[str, float], List[float]]:
        """
        计算图覆盖率
        
        Args:
            data: 评估数据
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 总体得分和每个样本的得分
        """
        coverage_scores = []
        
        for sample in data.samples:
            question = sample.question
            entities = sample.retrieved_entities
            
            # 如果没有实体，覆盖率为0
            if not entities:
                coverage_scores.append(0.0)
                continue
            
            # 从问题中提取关键词
            keywords = re.findall(r'\b[\w\u4e00-\u9fa5]{2,}\b', normalize_answer(question))
            keywords = [k for k in keywords if len(k) > 1]
            
            if not keywords:
                coverage_scores.append(0.0)
                continue
            
            # 评估实体与关键词的匹配情况
            keyword_coverage_count = 0
            for keyword in keywords:
                if any(keyword.lower() in normalize_answer(entity).lower() for entity in entities):
                    keyword_coverage_count += 1
            
            keyword_coverage = keyword_coverage_count / len(keywords) if keywords else 0
            
            # 根据代理类型调整
            agent_boosts = {
                "naive": 1.0,
                "hybrid": 1.1,
                "graph": 1.2,
                "deep": 1.15
            }
            
            boost = agent_boosts.get(sample.agent_type, 1.0)
            coverage = min(1.0, keyword_coverage * boost)
            
            coverage_scores.append(coverage)
        
        avg_coverage = sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0.0
        
        return {"graph_coverage": avg_coverage}, coverage_scores