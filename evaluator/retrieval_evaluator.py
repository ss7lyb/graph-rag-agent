from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field, asdict
import re
import time
import json

from evaluator.base import BaseEvaluator, BaseMetric
from evaluator.graph_metrics import GraphCoverageMetric, CommunityRelevanceMetric, SubgraphQualityMetric
from evaluator.preprocessing import clean_thinking_process, extract_references_from_answer, clean_references
from evaluator.utils import normalize_answer, extract_relationships_from_neo4j_response, extract_entities_from_neo4j_response, extract_referenced_entities, extract_referenced_relationships

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
        """
        更新系统回答并提取引用
        
        Args:
            answer: 原始系统回答
            agent_type: 代理类型
        """
        # 先清理思考过程，再提取引用数据
        if agent_type == "deep":
            answer = clean_thinking_process(answer)
            
        # 保存原始答案（包含引用数据）
        self.system_answer = answer
        
        if agent_type:
            self.agent_type = agent_type
            
        # 提取引用的实体和关系
        self.referenced_entities = list(extract_referenced_entities(answer))
        self.referenced_relationships = extract_referenced_relationships(answer)
    
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
    
    def calculate_metric(self, data: RetrievalEvaluationData) -> Tuple[Dict[str, float], List[float]]:
        """
        计算关系利用率 - 引用的关系在检索结果中的比例
        
        Args:
            data (RetrievalEvaluationData): 评估数据
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 总体得分和每个样本的得分
        """
        retrieved_rels = data.retrieved_relationships
        referenced_rels = data.referenced_relationships
        
        utilization_scores = []
        for retr_rels, ref_rels in zip(retrieved_rels, referenced_rels):
            if not ref_rels:
                # 没有引用关系，利用率为0
                utilization_scores.append(0.0)
                continue
            
            if not retr_rels:
                # 没有检索到关系，利用率为0
                utilization_scores.append(0.0)
                continue
            
            # 标准化处理
            retr_norm = [(normalize_answer(src), normalize_answer(rel), normalize_answer(dst)) 
                         for src, rel, dst in retr_rels]
            ref_norm = [(normalize_answer(src), normalize_answer(rel), normalize_answer(dst)) 
                        for src, rel, dst in ref_rels]
            
            # 计算利用率
            referenced = 0
            for r_src, r_rel, r_dst in ref_norm:
                for t_src, t_rel, t_dst in retr_norm:
                    # 检查源实体和目标实体以及关系类型是否匹配
                    if (r_src == t_src and r_dst == t_dst and r_rel == t_rel):
                        referenced += 1
                        break
            
            utilization = referenced / len(ref_norm) if ref_norm else 0.0
            utilization_scores.append(utilization)
        
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
        # 使用配置中提供的所有可能实体（如果有）
        self.all_possible_entities = config.get("possible_entities", [])
    
    def calculate_metric(self, data: RetrievalEvaluationData) -> Tuple[Dict[str, float], List[float]]:
        """
        计算实体覆盖率 - 根据关键词判断检索到的实体覆盖了多少关键概念
        
        Args:
            data (RetrievalEvaluationData): 评估数据
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 总体得分和每个样本的得分
        """
        questions = data.questions
        retrieved_entities = data.retrieved_entities
        
        coverage_scores = []
        
        for question, entities in zip(questions, retrieved_entities):
            # 从问题中提取关键词
            keywords = re.findall(r'\b[\w\u4e00-\u9fa5]{2,}\b', normalize_answer(question))
            keywords = [k for k in keywords if len(k) > 1]
            
            if not keywords:
                # 没有关键词，跳过
                coverage_scores.append(0.0)
                continue
                
            # 将检索到的实体名称连接为一个文本
            entities_text = " ".join([normalize_answer(e) for e in entities])
            
            # 计算关键词覆盖率
            covered = sum(1 for k in keywords if k.lower() in entities_text.lower())
            coverage = covered / len(keywords) if keywords else 0.0
            
            coverage_scores.append(coverage)
        
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
        """
        执行评估
        
        Args:
            data (RetrievalEvaluationData): 评估数据
            
        Returns:
            Dict[str, float]: 评估结果
        """
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
            
            # 更新样本
            sample.update_system_answer(answer, agent_name)
            sample.retrieval_time = time.time() - start_time
            
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
    
    def _extract_retrieval_data_from_logs(self, logs: List[Dict]) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """
        从执行日志中提取检索到的实体和关系
        
        Args:
            logs: 执行日志
            
        Returns:
            Tuple[List[str], List[Tuple[str, str, str]]]: 实体和关系
        """
        entities = []
        relationships = []
        
        # 查找检索相关的日志条目
        retrieval_logs = [
            log for log in logs 
            if log.get('node', '').lower() in ['retrieve', 'retrieval', 'search']
        ]
        
        for log in retrieval_logs:
            # 检查输出内容
            output = log.get('output', '')
            if isinstance(output, str):
                # 提取实体和关系
                extracted_entities = extract_entities_from_neo4j_response(output)
                extracted_rels = extract_relationships_from_neo4j_response(output)
                
                entities.extend(extracted_entities)
                relationships.extend(extracted_rels)
            elif isinstance(output, dict):
                # 检查是否有结构化数据
                if 'entities' in output:
                    entities.extend(output['entities'])
                if 'relationships' in output:
                    relationships.extend(output['relationships'])
        
        # 去重
        unique_entities = list(set(entities))
        
        # 关系去重
        seen_rels = set()
        unique_rels = []
        for rel in relationships:
            rel_key = (rel[0], rel[1], rel[2])
            if rel_key not in seen_rels:
                seen_rels.add(rel_key)
                unique_rels.append(rel)
        
        return unique_entities, unique_rels
    
    def _get_relevant_graph_data(self, question: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """
        从Neo4j获取与问题相关的实体和关系
        
        Args:
            question: 问题
            
        Returns:
            Tuple[List[str], List[Tuple[str, str, str]]]: 实体和关系
        """
        if not self.neo4j_client:
            return [], []
            
        # 提取问题中的关键词
        question_words = re.findall(r'\b[\w\u4e00-\u9fa5]{2,}\b', normalize_answer(question))
        question_words = [w for w in question_words if len(w) > 1]
        
        entities = []
        relationships = []
        
        try:
            # 查询与关键词匹配的实体
            entity_query = """
            MATCH (n:__Entity__)
            WHERE any(word IN $keywords WHERE n.id CONTAINS word OR n.description CONTAINS word)
            RETURN n.id AS id
            LIMIT 10
            """
            
            # 查询与这些实体相关的关系
            relationship_query = """
            MATCH (a:__Entity__)-[r]->(b:__Entity__)
            WHERE a.id IN $entity_ids OR b.id IN $entity_ids
            RETURN a.id AS source, type(r) AS relation, b.id AS target
            LIMIT 20
            """
            
            # 执行查询
            entity_result = self.neo4j_client.execute_query(entity_query, {"keywords": question_words})

            if entity_result.records:
                for record in entity_result.records:
                    if 'id' in record:
                        entity_id = record['id']
                        if entity_id:
                            entities.append(entity_id)

            rel_result = self.neo4j_client.execute_query(relationship_query, {"entity_ids": entities})
            if rel_result.records:
                for record in rel_result.records:
                    if 'source' in record and 'relation' in record and 'target' in record:
                        source = record['source']
                        relation = record['relation']
                        target = record['target']
                        if source and relation and target:
                            relationships.append((source, relation, target))
        except Exception as e:
            print(f"从Neo4j获取图数据时出错: {e}")
        
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
