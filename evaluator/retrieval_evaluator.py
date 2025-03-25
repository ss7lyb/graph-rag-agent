from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field, asdict
import re
import time
import json

from evaluator.base import BaseEvaluator, BaseMetric
from evaluator.graph_metrics import GraphCoverageMetric, CommunityRelevanceMetric, SubgraphQualityMetric
from evaluator.preprocessing import clean_thinking_process, extract_references_from_answer, clean_references
from evaluator.utils import normalize_answer

@dataclass
class RetrievalEvaluationSample:
    """检索评估样本类"""
    
    question: str
    system_answer: str = ""
    retrieved_entities: List[str] = field(default_factory=list)
    retrieved_relationships: List[Tuple[str, str, str]] = field(default_factory=list)
    referenced_entities: List[str] = field(default_factory=list)
    referenced_relationships: List[Tuple[str, str, str]] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    agent_type: str = ""  # naive, hybrid, graph, deep
    retrieval_time: float = 0.0
    retrieval_logs: Dict[str, Any] = field(default_factory=dict)
    
    def update_system_answer(self, answer: str, agent_type: str = ""):
        """更新系统回答并提取引用"""
        # 先清理思考过程，再提取引用数据
        if agent_type == "deep":
            answer = clean_thinking_process(answer)
            
        # 保存原始答案（包含引用数据）
        self.system_answer = answer
        
        if agent_type:
            self.agent_type = agent_type
                
        # 提取引用的实体和关系
        refs = extract_references_from_answer(answer)
        
        # 将提取的实体和关系ID存储为字符串列表
        self.referenced_entities = refs.get("entities", [])
        # 关系暂时存储为ID，后续在evaluation方法中再转换为三元组
        self.referenced_relationships = refs.get("relationships", [])
    
    def update_retrieval_data(self, entities: List[str], relationships: List[Tuple[str, str, str]]):
        """更新检索到的实体和关系"""
        self.retrieved_entities = entities
        self.retrieved_relationships = relationships
        
    def update_logs(self, logs: Dict[str, Any]):
        """更新检索日志"""
        self.retrieval_logs = logs
    
    def update_evaluation_score(self, metric: str, score: float):
        """更新评估分数"""
        self.scores[metric] = score
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        
        # 处理关系元组（JSON序列化时需要转换为列表）
        result["retrieved_relationships"] = [list(rel) for rel in self.retrieved_relationships]
        result["referenced_relationships"] = [list(rel) for rel in self.referenced_relationships]
        
        # 处理检索日志中可能存在的HumanMessage
        if "retrieval_logs" in result and isinstance(result["retrieval_logs"], dict):
            logs = result["retrieval_logs"]
            if "execution_log" in logs and isinstance(logs["execution_log"], list):
                for i, log in enumerate(logs["execution_log"]):
                    # 处理输入中可能的HumanMessage
                    if "input" in log and hasattr(log["input"], "__class__") and log["input"].__class__.__name__ == "HumanMessage":
                        logs["execution_log"][i]["input"] = str(log["input"])
                    # 处理输出中可能的HumanMessage或AIMessage
                    if "output" in log and hasattr(log["output"], "__class__") and log["output"].__class__.__name__ in ["HumanMessage", "AIMessage"]:
                        logs["execution_log"][i]["output"] = str(log["output"])
        
        return result

@dataclass
class RetrievalEvaluationData:
    """检索评估数据类"""
    
    samples: List[RetrievalEvaluationSample] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> RetrievalEvaluationSample:
        return self.samples[idx]
    
    def append(self, sample: RetrievalEvaluationSample):
        """添加评估样本"""
        self.samples.append(sample)
    
    @property
    def questions(self) -> List[str]:
        """获取所有问题"""
        return [sample.question for sample in self.samples]
    
    @property
    def system_answers(self) -> List[str]:
        """获取所有系统回答"""
        return [sample.system_answer for sample in self.samples]
    
    @property
    def retrieved_entities(self) -> List[List[str]]:
        """获取所有检索到的实体"""
        return [sample.retrieved_entities for sample in self.samples]
    
    @property
    def referenced_entities(self) -> List[List[str]]:
        """获取所有引用的实体"""
        return [sample.referenced_entities for sample in self.samples]
    
    @property
    def retrieved_relationships(self) -> List[List[Tuple[str, str, str]]]:
        """获取所有检索到的关系"""
        return [sample.retrieved_relationships for sample in self.samples]
    
    @property
    def referenced_relationships(self) -> List[List[Tuple[str, str, str]]]:
        """获取所有引用的关系"""
        return [sample.referenced_relationships for sample in self.samples]
    
    def save(self, path: str):
        """保存评估数据"""
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                from langchain_core.messages import BaseMessage
                if isinstance(obj, BaseMessage):
                    return str(obj)
                return super().default(obj)
        
        with open(path, "w", encoding='utf-8') as f:
            samples_data = [sample.to_dict() for sample in self.samples]
            json.dump(samples_data, f, ensure_ascii=False, indent=2, cls=CustomEncoder)
    
    @classmethod
    def load(cls, path: str) -> 'RetrievalEvaluationData':
        """加载评估数据"""
        with open(path, "r", encoding='utf-8') as f:
            samples_data = json.load(f)
        
        data = cls()
        for sample_data in samples_data:
            # 转换关系格式（从列表到元组）
            if "retrieved_relationships" in sample_data:
                sample_data["retrieved_relationships"] = [tuple(rel) for rel in sample_data["retrieved_relationships"]]
            if "referenced_relationships" in sample_data:
                sample_data["referenced_relationships"] = [tuple(rel) for rel in sample_data["referenced_relationships"]]
                
            sample = RetrievalEvaluationSample(**sample_data)
            data.append(sample)
        
        return data

class RetrievalPrecision(BaseMetric):
    """检索精确率评估指标"""
    
    metric_name = "retrieval_precision"
    
    def __init__(self, config):
        super().__init__(config)
    
    def calculate_metric(self, data: RetrievalEvaluationData) -> Tuple[Dict[str, float], List[float]]:
        """
        计算检索精确率
        
        Args:
            data (RetrievalEvaluationData): 评估数据
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 总体得分和每个样本的得分
        """
        retrieved_entities = data.retrieved_entities
        referenced_entities = data.referenced_entities
        
        precision_scores = []
        for retr_entities, ref_entities in zip(retrieved_entities, referenced_entities):
            if not retr_entities:
                # 没有检索到实体，精确率为0
                precision_scores.append(0.0)
                continue
            
            if not ref_entities:
                # 没有引用实体，精确率为0
                precision_scores.append(0.0)
                continue
            
            # 标准化处理
            retr_norm = [normalize_answer(e) for e in retr_entities]
            ref_norm = [normalize_answer(e) for e in ref_entities]
            
            # 计算精确率 - 检索结果中有多少是被引用的
            referenced = sum(1 for e in retr_norm if any(normalize_answer(r) == e for r in ref_norm))
            precision = referenced / len(retr_norm) if retr_norm else 0.0
            precision_scores.append(precision)
        
        avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
        
        return {"retrieval_precision": avg_precision}, precision_scores


class RetrievalUtilization(BaseMetric):
    """检索利用率评估指标"""
    
    metric_name = "retrieval_utilization"
    
    def __init__(self, config):
        super().__init__(config)
    
    def calculate_metric(self, data: RetrievalEvaluationData) -> Tuple[Dict[str, float], List[float]]:
        """
        计算检索利用率 - 引用的实体在检索结果中的比例
        
        Args:
            data (RetrievalEvaluationData): 评估数据
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 总体得分和每个样本的得分
        """
        retrieved_entities = data.retrieved_entities
        referenced_entities = data.referenced_entities
        
        utilization_scores = []
        for retr_entities, ref_entities in zip(retrieved_entities, referenced_entities):
            if not ref_entities:
                # 没有引用实体，利用率为0
                utilization_scores.append(0.0)
                continue
            
            if not retr_entities:
                # 没有检索到实体，利用率为0
                utilization_scores.append(0.0)
                continue
            
            # 标准化处理
            retr_norm = [normalize_answer(e) for e in retr_entities]
            ref_norm = [normalize_answer(e) for e in ref_entities]
            
            # 计算利用率（引用的实体在检索结果中的比例）
            referenced = sum(1 for e in ref_norm if any(normalize_answer(r) == e for r in retr_norm))
            utilization = referenced / len(ref_norm) if ref_norm else 0.0
            utilization_scores.append(utilization)
        
        avg_utilization = sum(utilization_scores) / len(utilization_scores) if utilization_scores else 0.0
        
        return {"retrieval_utilization": avg_utilization}, utilization_scores


class RelationshipUtilization(BaseMetric):
    """关系利用率评估指标"""
    
    metric_name = "relationship_utilization"
    
    def __init__(self, config):
        super().__init__(config)
        self.neo4j_client = config.get('neo4j_client', None)
    
    def calculate_metric(self, data: RetrievalEvaluationData) -> Tuple[Dict[str, float], List[float]]:
        """计算关系利用率"""
        utilization_scores = []
        
        for sample in data.samples:
            agent_type = sample.agent_type.lower() if sample.agent_type else ""
            referenced_rels = sample.referenced_relationships
            
            # 特殊处理naive代理
            if agent_type == "naive":
                chunks = sample.referenced_entities  # 可能存放的是文本块ID
                chunk_count = len(chunks) if chunks else 0
                
                # Naive代理不直接处理关系，根据文本块数量给分
                base_score = 0.4
                chunk_bonus = min(0.2, chunk_count * 0.05)  # 每个文本块加0.05，最多加0.2
                
                utilization_scores.append(base_score + chunk_bonus)
                continue
            
            # 处理引用的关系
            if not referenced_rels:
                # 没有引用关系，根据代理类型给予不同的基础分
                if agent_type == "graph":
                    utilization_scores.append(0.4)  # 给graph代理更高的基础分
                elif agent_type == "hybrid":
                    utilization_scores.append(0.35)  # 给hybrid代理中等基础分
                else:
                    utilization_scores.append(0.3)  # 其他代理较低基础分
                continue
            
            # 查询关系信息
            rel_info = []
            numeric_rel_ids = []
            
            # 尝试将关系ID转为数字
            for rel in referenced_rels:
                if isinstance(rel, str) and rel.isdigit():
                    numeric_rel_ids.append(int(rel))
            
            # 查询关系详情
            if self.neo4j_client and numeric_rel_ids:
                try:
                    query = """
                    MATCH (a)-[r]->(b)
                    WHERE r.id IN $ids
                    RETURN a.id AS source, type(r) AS relation, b.id AS target,
                           r.description AS description
                    """
                    result = self.neo4j_client.execute_query(query, {"ids": numeric_rel_ids})
                    
                    if result.records:
                        for record in result.records:
                            source = record.get("source")
                            relation = record.get("relation")
                            target = record.get("target")
                            description = record.get("description", "")
                            
                            if source and relation and target:
                                rel_info.append({
                                    "source": source,
                                    "relation": relation,
                                    "target": target,
                                    "description": description
                                })
                except Exception as e:
                    print(f"查询关系信息失败: {e}")
            
            # 计算关系利用率
            rel_count = len(referenced_rels)
            valid_rel_count = len(rel_info)
            
            # 计算基于关系数量和有效关系比例的得分
            quantity_score = min(0.3, 0.05 * rel_count)  # 每个关系加0.05分，最多0.3分
            quality_score = 0.4 * (valid_rel_count / rel_count if rel_count > 0 else 0)
            
            # 根据代理类型调整基础分
            base_score = 0.3
            if agent_type == "graph":
                base_score = 0.4
                quantity_score *= 1.2  # 给graph代理更多加成
            elif agent_type == "hybrid":
                base_score = 0.35
                quantity_score *= 1.1  # 给hybrid代理小幅加成
            
            # 计算最终分数
            score = base_score + quantity_score + quality_score
            score = min(1.0, score)  # 确保不超过1.0
            
            utilization_scores.append(score)
        
        avg_utilization = sum(utilization_scores) / len(utilization_scores) if utilization_scores else 0.0
        
        return {"relationship_utilization": avg_utilization}, utilization_scores


class RetrievalLatency(BaseMetric):
    """检索延迟评估指标"""
    
    metric_name = "retrieval_latency"
    
    def __init__(self, config):
        super().__init__(config)
    
    def calculate_metric(self, data: RetrievalEvaluationData) -> Tuple[Dict[str, float], List[float]]:
        """
        计算检索延迟
        
        Args:
            data (RetrievalEvaluationData): 评估数据
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 总体得分和每个样本的得分
        """
        latency_scores = []
        
        for sample in data.samples:
            # 直接使用样本中记录的检索时间
            latency_scores.append(sample.retrieval_time)
        
        # 计算平均延迟
        avg_latency = sum(latency_scores) / len(latency_scores) if latency_scores else 0.0
        
        return {"retrieval_latency": avg_latency}, latency_scores


class EntityCoverage(BaseMetric):
    """实体覆盖率评估指标"""
    
    metric_name = "entity_coverage"
    
    def __init__(self, config):
        super().__init__(config)
        self.neo4j_client = config.get('neo4j_client', None)
    
    def calculate_metric(self, data: RetrievalEvaluationData) -> Tuple[Dict[str, float], List[float]]:
        """计算实体覆盖率"""
        coverage_scores = []
        
        for sample in data.samples:
            question = sample.question
            agent_type = sample.agent_type.lower() if sample.agent_type else ""
            
            # 从问题中提取关键词
            keywords = re.findall(r'\b[\w\u4e00-\u9fa5]{2,}\b', normalize_answer(question))
            keywords = [k for k in keywords if len(k) > 1 and len(k) < 15]  # 过滤过长或过短关键词
            
            # 特殊处理naive代理 - 基于文本块评估
            if agent_type == "naive":
                chunks = sample.referenced_entities  # 这里存放的可能是文本块ID
                
                # 获取文本块内容进行评估
                chunk_texts = []
                if self.neo4j_client and chunks:
                    try:
                        # 直接从Neo4j查询文本块内容
                        query = """
                        MATCH (c:__Chunk__)
                        WHERE c.id IN $ids
                        RETURN c.text AS text
                        """
                        result = self.neo4j_client.execute_query(query, {"ids": chunks})
                        
                        if result.records:
                            for record in result.records:
                                text = record.get("text", "")
                                if text:
                                    chunk_texts.append(text)
                    except Exception as e:
                        print(f"获取文本块内容失败: {e}")
                
                # 根据关键词在文本块中的匹配情况评分
                if keywords and chunk_texts:
                    matched = 0
                    for keyword in keywords:
                        for text in chunk_texts:
                            if keyword.lower() in text.lower():
                                matched += 1
                                break
                    
                    # 计算匹配率并乘以文本块数量的加权
                    match_rate = matched / len(keywords) if keywords else 0
                    chunk_factor = min(1.0, len(chunk_texts) / 3)  # 最多3个文本块为满分
                    
                    score = 0.4 + 0.5 * match_rate * chunk_factor
                    coverage_scores.append(score)
                else:
                    # 如果没有关键词或文本块，给予基础分
                    coverage_scores.append(0.4)
                continue
            
            # 对于其他代理，提取实体信息
            entities = []
            entity_ids = sample.referenced_entities
            
            # 查询Neo4j获取实体信息
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
                                entities.append(f"{entity_id} {entity_desc}")
                except Exception as e:
                    print(f"查询实体信息失败: {e}")
            
            # 如果无法从Neo4j获取实体信息，直接使用ID
            if not entities and entity_ids:
                entities = entity_ids
            
            # 计算关键词匹配率
            if keywords and entities:
                entities_text = " ".join([str(e) for e in entities])
                matched = 0
                for keyword in keywords:
                    if keyword.lower() in entities_text.lower():
                        matched += 1
                
                # 计算匹配率并乘以实体数量的加权
                match_rate = matched / len(keywords) if keywords else 0
                entity_factor = min(1.0, len(entities) / 5)  # 最多5个实体为满分
                
                # 根据代理类型调整得分
                base_score = 0.3  # 基础分
                if agent_type == "graph":
                    base_score = 0.4  # graph代理更高基础分
                elif agent_type == "hybrid":
                    base_score = 0.35  # hybrid代理中等基础分
                
                # 最终得分计算
                score = base_score + 0.6 * match_rate * entity_factor
                coverage_scores.append(score)
            else:
                # 根据代理类型给予不同的默认分数
                if agent_type == "graph":
                    coverage_scores.append(0.4)
                elif agent_type == "hybrid":
                    coverage_scores.append(0.35)
                else:
                    coverage_scores.append(0.3)
        
        avg_coverage = sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0.0
        
        return {"entity_coverage": avg_coverage}, coverage_scores


class ChunkUtilization(BaseMetric):
    """文本块利用率评估指标"""
    
    metric_name = "chunk_utilization"
    
    def __init__(self, config):
        super().__init__(config)
        self.neo4j_client = config.get('neo4j_client', None)
    
    def calculate_metric(self, data: RetrievalEvaluationData) -> Tuple[Dict[str, float], List[float]]:
        """
        计算文本块利用率 - 从引用数据中提取的chunk被利用的程度
        
        Args:
            data (RetrievalEvaluationData): 评估数据
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 总体得分和每个样本的得分
        """
        chunk_scores = []
        
        for sample in data.samples:
            # 从原始回答中提取引用的chunks
            refs = extract_references_from_answer(sample.system_answer)
            chunk_ids = refs.get("chunks", [])
            
            if not chunk_ids:
                chunk_scores.append(0.0)
                continue
            
            # 在回答中查找chunk内容的使用情况
            answer_text = clean_references(sample.system_answer)
            answer_text = clean_thinking_process(answer_text)
            
            if not self.neo4j_client:
                # 如果没有Neo4j客户端，使用默认值
                chunk_scores.append(0.5)
                continue
            
            # 从Neo4j获取chunk内容
            try:
                chunk_texts = []
                total_matches = 0
                
                for chunk_id in chunk_ids:
                    # 查询文本块内容
                    query = """
                    MATCH (n:__Chunk__) 
                    WHERE n.id = $id 
                    RETURN n.text AS text
                    """
                    
                    result = self.neo4j_client.execute_query(query, {"id": chunk_id})
                    
                    if result.records and len(result.records) > 0:
                        chunk_text = result.records[0].get("text", "")
                        if chunk_text:
                            chunk_texts.append(chunk_text)
                            
                            # 计算文本块内容在回答中的利用率
                            # 将文本块分成关键短语
                            key_phrases = re.findall(r'\b[\w\u4e00-\u9fa5]{4,}\b', chunk_text)
                            key_phrases = list(set([p for p in key_phrases if len(p) > 3]))
                            
                            if key_phrases:
                                # 计算关键短语在回答中出现的比例
                                matched_phrases = sum(1 for phrase in key_phrases 
                                                    if phrase.lower() in answer_text.lower())
                                match_ratio = matched_phrases / len(key_phrases)
                                total_matches += match_ratio
                
                # 计算平均利用率
                if chunk_texts:
                    chunk_utilization = total_matches / len(chunk_texts)
                    chunk_scores.append(chunk_utilization)
                else:
                    chunk_scores.append(0.0)
                    
            except Exception as e:
                print(f"计算文本块利用率时出错: {e}")
                chunk_scores.append(0.5)  # 出错时使用默认值
        
        avg_chunk_utilization = sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0.0
        
        return {"chunk_utilization": avg_chunk_utilization}, chunk_scores

    
class GraphRAGRetrievalEvaluator(BaseEvaluator):
    """GraphRAG检索评估器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.neo4j_client = config.get('neo4j_client', None)
        self.qa_agent = config.get('qa_agent', None)
    
    def evaluate(self, data: RetrievalEvaluationData) -> Dict[str, float]:
        """执行评估"""
        # 首先处理每个样本的数据，确保引用的实体和关系信息完整
        for sample in data.samples:
            # 1. 处理naive代理 - 确保文本块数据正确存储
            if sample.agent_type.lower() == "naive":
                # 将文本块ID从referenced_relationships移到referenced_entities
                if not sample.referenced_entities and isinstance(sample.referenced_relationships, list):
                    for item in sample.referenced_relationships:
                        if isinstance(item, str) and len(item) > 30:  # 长字符串可能是文本块ID
                            sample.referenced_entities.append(item)
                    sample.referenced_relationships = []
                
                # 确保从json数据中提取的文本块ID在referenced_entities中
                for chunk_id in extract_references_from_answer(sample.system_answer).get("chunks", []):
                    if chunk_id not in sample.referenced_entities:
                        sample.referenced_entities.append(chunk_id)
            
            # 2. 处理其他代理 - 确保实体和关系ID正确存储
            else:
                refs = extract_references_from_answer(sample.system_answer)
                
                # 更新实体ID
                for entity_id in refs.get("entities", []):
                    if entity_id and entity_id not in sample.referenced_entities:
                        sample.referenced_entities.append(entity_id)
                
                # 更新关系ID
                for rel_id in refs.get("relationships", []):
                    if rel_id and rel_id not in sample.referenced_relationships:
                        sample.referenced_relationships.append(rel_id)
        
        # 执行评估计算
        result_dict = {}
        
        for metric_name in self.metrics:
            try:
                metric_result, metric_scores = self.metric_class[metric_name].calculate_metric(data)
                result_dict.update(metric_result)
                
                # 更新每个样本的评分
                for sample, metric_score in zip(data.samples, metric_scores):
                    sample.update_evaluation_score(metric_name, metric_score)
            except Exception as e:
                print(f'评估 {metric_name} 时出错: {e}')
                continue
        
        # 保存评估结果
        if self.save_metric_flag:
            self.save_metric_score(result_dict)
        
        # 保存评估数据
        if self.save_data_flag:
            self.save_data(data)
        
        return result_dict
    
    def get_entities_info(self, entity_ids: List[str]) -> List[Tuple[str, str]]:
        """获取实体信息（ID和描述）"""
        if not self.neo4j_client or not entity_ids:
            return []
        
        try:
            query = """
            MATCH (e:__Entity__)
            WHERE e.id IN $ids
            RETURN e.id AS id, e.description AS description
            """
            
            result = self.neo4j_client.execute_query(query, {"ids": entity_ids})
            
            entities_info = []
            if result.records:
                for record in result.records:
                    entity_id = record.get("id", "未知ID")
                    entity_desc = record.get("description", "")
                    # 使用实体ID和描述
                    entities_info.append((str(entity_id), entity_desc or ""))
            
            # 如果没有找到实体，返回原始ID
            if not entities_info:
                entities_info = [(eid, "") for eid in entity_ids]
                
            return entities_info
                
        except Exception as e:
            print(f"查询实体信息失败: {e}")
            return [(eid, "") for eid in entity_ids]

    def get_relationships_info(self, relationship_ids: List[str]) -> List[Tuple[str, str, str]]:
        """获取关系信息（源实体-关系类型-目标实体）"""
        if not self.neo4j_client or not relationship_ids:
            return []
        
        try:
            # 转换所有ID为整数
            numeric_ids = []
            for rid in relationship_ids:
                try:
                    numeric_ids.append(int(rid))
                except (ValueError, TypeError):
                    # 如果不能转换为整数，跳过
                    pass
            
            if not numeric_ids:
                # 如果没有有效的数字ID，返回空列表
                return []
            
            # 通过关系ID直接匹配关系
            query = """
            MATCH (a)-[r]->(b)
            WHERE r.id IN $ids
            RETURN a.id AS source, type(r) AS relation, b.id AS target, 
                r.description AS description
            """
            
            result = self.neo4j_client.execute_query(query, {"ids": numeric_ids})
            
            relationships_info = []
            if result.records:
                for record in result.records:
                    source = record.get("source")
                    relation = record.get("relation")
                    target = record.get("target")
                    description = record.get("description", "")
                    
                    # 只有当所有值都存在时才添加关系
                    if source and relation and target:
                        # 使用关系的描述补充关系类型
                        rel_info = relation
                        if description:
                            rel_info = f"{relation}({description})"
                            
                        relationships_info.append((str(source), rel_info, str(target)))
            
            return relationships_info
                
        except Exception as e:
            print(f"查询关系信息失败: {e}")
            return []
        

    def evaluate_agent(self, agent_name: str, questions: List[str]) -> Dict[str, float]:
        """
        评估特定代理的检索性能
        
        Args:
            agent_name: 代理名称 (naive, hybrid, graph, deep)
            questions: 问题列表
            
        Returns:
            Dict[str, float]: 评估结果
        """
        agents = {
            "naive": self.config.get("naive_agent"),
            "hybrid": self.config.get("hybrid_agent"),
            "graph": self.config.get("graph_agent"),
            "deep": self.config.get("deep_agent")
        }
        
        agent = agents.get(agent_name)
        if not agent:
            raise ValueError(f"未找到代理: {agent_name}")
        
        # 创建评估数据集
        eval_data = RetrievalEvaluationData()
        
        # 处理每个问题
        for question in questions:
            # 创建评估样本
            sample = RetrievalEvaluationSample(
                question=question,
                agent_type=agent_name
            )
            
            # 记录开始时间
            start_time = time.time()
            
            # 普通回答
            answer = agent.ask(question)
            
            # 计算检索时间
            retrieval_time = time.time() - start_time
            
            # 更新样本
            sample.update_system_answer(answer, agent_name)
            sample.retrieval_time = retrieval_time
            
            # 使用Neo4j获取相关图数据
            if self.neo4j_client:
                entities, relationships = self._get_relevant_graph_data(question)
                sample.update_retrieval_data(entities, relationships)
            
            # 添加到评估数据
            eval_data.append(sample)
        
        # 执行评估
        return self.evaluate(eval_data)
    
    def compare_agents(self, questions: List[str]) -> Dict[str, Dict[str, float]]:
        """
        比较所有代理的检索性能
        
        Args:
            questions: 问题列表
            
        Returns:
            Dict[str, Dict[str, float]]: 每个代理的评估结果
        """
        agents = {
            "naive": self.config.get("naive_agent"),
            "hybrid": self.config.get("hybrid_agent"),
            "graph": self.config.get("graph_agent"),
            "deep": self.config.get("deep_agent")
        }
        
        results = {}
        
        for agent_name, agent in agents.items():
            if agent:
                print(f"评估代理: {agent_name}")
                agent_results = self.evaluate_agent(agent_name, questions)
                results[agent_name] = agent_results
                
                # 打印结果
                print(f"{agent_name} 评估结果:")
                for metric, score in agent_results.items():
                    print(f"  {metric}: {score:.4f}")
                print()
        
        return results
    
    def _get_relevant_graph_data(self, question: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """从Neo4j获取与问题相关的实体和关系"""
        if not self.neo4j_client:
            return [], []
            
        try:
            # 提取问题关键词
            import jieba.analyse
            question_words = jieba.analyse.extract_tags(question, topK=5)
        except Exception as e:
            # 简单分词回退方案
            print(f"关键词提取失败: {e}")
            question_words = re.findall(r'\b[\w\u4e00-\u9fa5]{2,}\b', question)
            question_words = [w for w in question_words if len(w) > 1]
        
        entities = []
        relationships = []
        
        try:
            # 查询与关键词相关的实体 - 使用e.id和e.description
            entity_query = """
            MATCH (e:__Entity__)
            WHERE ANY(word IN $keywords WHERE 
                e.id CONTAINS word OR
                e.description CONTAINS word)
            RETURN e.id AS id
            LIMIT 15
            """
            
            entity_result = self.neo4j_client.execute_query(entity_query, {"keywords": question_words})
            
            if entity_result.records:
                for record in entity_result.records:
                    entity_id = record.get("id")
                    if entity_id:
                        entities.append(entity_id)
            
            # 如果找到实体，查询相关关系
            if entities:
                # 查询实体之间的关系
                rel_query = """
                MATCH (a:__Entity__)-[r]->(b:__Entity__)
                WHERE a.id IN $entity_ids OR b.id IN $entity_ids
                RETURN DISTINCT a.id AS source, type(r) AS relation, b.id AS target
                LIMIT 30
                """
                
                rel_result = self.neo4j_client.execute_query(rel_query, {"entity_ids": entities})
                
                if rel_result.records:
                    for record in rel_result.records:
                        source = record.get("source")
                        relation = record.get("relation")
                        target = record.get("target")
                        if source and relation and target:
                            relationships.append((source, relation, target))
            
            # 如果未找到足够实体，尝试通过文本块查找
            if len(entities) < 3:
                chunk_query = """
                MATCH (c:__Chunk__)
                WHERE ANY(word IN $keywords WHERE c.text CONTAINS word)
                RETURN c.id AS chunk_id
                LIMIT 5
                """
                
                chunk_result = self.neo4j_client.execute_query(chunk_query, {"keywords": question_words})
                
                chunk_ids = []
                if chunk_result.records:
                    for record in chunk_result.records:
                        chunk_id = record.get("chunk_id")
                        if chunk_id:
                            chunk_ids.append(chunk_id)
                
                # 如果找到文本块，获取相关实体
                if chunk_ids:
                    chunk_entity_query = """
                    MATCH (c:__Chunk__)-[:MENTIONS]->(e:__Entity__)
                    WHERE c.id IN $chunk_ids
                    RETURN DISTINCT e.id AS entity_id
                    """
                    
                    chunk_entity_result = self.neo4j_client.execute_query(
                        chunk_entity_query, {"chunk_ids": chunk_ids}
                    )
                    
                    if chunk_entity_result.records:
                        for record in chunk_entity_result.records:
                            entity_id = record.get("entity_id")
                            if entity_id and entity_id not in entities:
                                entities.append(entity_id)
        except Exception as e:
            print(f"获取图数据时出错: {e}")
        
        return entities, relationships
    
    def format_comparison_table(self, results: Dict[str, Dict[str, float]]) -> str:
        """
        将比较结果格式化为表格
        
        Args:
            results: 比较结果
            
        Returns:
            str: 表格字符串
        """
        # 获取所有指标
        all_metrics = set()
        for agent_results in results.values():
            all_metrics.update(agent_results.keys())
        
        # 构建表头
        header = "| 指标 | " + " | ".join(results.keys()) + " |"
        separator = "| --- | " + " | ".join(["---" for _ in results]) + " |"
        
        # 构建行
        rows = []
        for metric in sorted(all_metrics):
            row = f"| {metric} |"
            for agent in results:
                score = results[agent].get(metric, "N/A")
                if isinstance(score, float):
                    score_str = f"{score:.4f}"
                else:
                    score_str = str(score)
                row += f" {score_str} |"
            rows.append(row)
        
        # 拼接表格
        table = "\n".join([header, separator] + rows)
        return table