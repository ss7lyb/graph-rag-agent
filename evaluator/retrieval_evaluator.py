import os
import json
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict

from evaluator.base import BaseEvaluator, BaseMetric
from evaluator.utils import (
    normalize_answer,
    extract_entities_from_neo4j_response,
    extract_relationships_from_neo4j_response,
    extract_referenced_entities,
    extract_referenced_relationships,
    extract_json_from_text
)


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
    retrieval_logs: Dict[str, Any] = field(default_factory=dict)
    
    def update_answer(self, answer: str):
        """更新系统回答"""
        self.system_answer = answer
        # 提取答案中引用的实体和关系
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
        return asdict(self)


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
        with open(path, "w", encoding='utf-8') as f:
            json.dump([sample.to_dict() for sample in self.samples], f, ensure_ascii=False, indent=2)
    
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
            
            # 计算精确率
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
        计算检索利用率
        
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
        计算关系利用率
        
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


class FactualConsistency(BaseMetric):
    """事实一致性评估指标"""
    
    metric_name = "factual_consistency"
    
    def __init__(self, config):
        super().__init__(config)
        self.use_llm = config.get('use_llm', True)
        
        if self.use_llm:
            from model.get_models import get_llm_model
            self.llm = get_llm_model()
    
    def _assess_consistency_simple(self, answer: str, entities: List[str], relationships: List[Tuple[str, str, str]]) -> float:
        """
        使用简单方法评估事实一致性
        
        Args:
            answer (str): 系统回答
            entities (List[str]): 检索到的实体
            relationships (List[Tuple[str, str, str]]): 检索到的关系
            
        Returns:
            float: 一致性得分 (0-1)
        """
        if not answer:
            return 0.0
            
        # 提取回答中引用的实体和关系
        referenced_entities = extract_referenced_entities(answer)
        referenced_relationships = extract_referenced_relationships(answer)
        
        # 标准化处理
        entities_norm = [normalize_answer(e) for e in entities]
        ref_entities_norm = [normalize_answer(e) for e in referenced_entities]
        
        # 计算实体一致性
        entity_consistency = 0.0
        if ref_entities_norm:
            consistent_entities = sum(1 for e in ref_entities_norm if any(normalize_answer(r) == e for r in entities_norm))
            entity_consistency = consistent_entities / len(ref_entities_norm)
        
        # 标准化处理关系
        rels_norm = [(normalize_answer(src), normalize_answer(rel), normalize_answer(dst)) 
                     for src, rel, dst in relationships]
        ref_rels_norm = [(normalize_answer(src), normalize_answer(rel), normalize_answer(dst)) 
                         for src, rel, dst in referenced_relationships]
        
        # 计算关系一致性
        rel_consistency = 0.0
        if ref_rels_norm:
            consistent_rels = 0
            for r_src, r_rel, r_dst in ref_rels_norm:
                for t_src, t_rel, t_dst in rels_norm:
                    if (r_src == t_src and r_dst == t_dst and r_rel == t_rel):
                        consistent_rels += 1
                        break
            
            rel_consistency = consistent_rels / len(ref_rels_norm)
        
        # 综合一致性得分，实体和关系各占50%权重
        if ref_entities_norm or ref_rels_norm:
            consistency = 0.0
            weights = 0.0
            
            if ref_entities_norm:
                consistency += 0.5 * entity_consistency
                weights += 0.5
                
            if ref_rels_norm:
                consistency += 0.5 * rel_consistency
                weights += 0.5
                
            return consistency / weights if weights > 0 else 0.0
        
        return 0.0  # 如果没有引用任何内容，则一致性为0
    
    def _assess_consistency_llm(self, answer: str, entities: List[str], relationships: List[Tuple[str, str, str]]) -> float:
        """
        使用LLM评估事实一致性
        
        Args:
            answer (str): 系统回答
            entities (List[str]): 检索到的实体
            relationships (List[Tuple[str, str, str]]): 检索到的关系
            
        Returns:
            float: 一致性得分 (0-1)
        """
        # 格式化实体和关系
        entities_text = "\n".join([f"- {entity}" for entity in entities])
        
        relationships_text = ""
        for src, rel, dst in relationships:
            relationships_text += f"- {src} --[{rel}]--> {dst}\n"
        
        prompt = f"""评估以下回答在多大程度上与提供的图数据保持一致，没有添加未在图数据中的信息。
        
回答:
{answer}

图数据中的实体:
{entities_text}

图数据中的关系:
{relationships_text}

请基于以下标准评分(0到1):
- 0.0: 完全不一致，回答中的主要信息不在图数据中
- 0.25: 轻微一致，少数信息来自图数据，但大部分是添加的
- 0.5: 部分一致，约一半的信息来自图数据
- 0.75: 大部分一致，主要信息来自图数据，有少量添加
- 1.0: 完全一致，回答中的所有主要信息都来自图数据

仅返回一个数值得分，不需要解释。示例：0.75
"""
        
        try:
            response = self.llm.invoke(prompt)
            
            # 提取数值得分
            if hasattr(response, 'content'):
                score_text = response.content
            else:
                score_text = str(response)
                
            # 尝试提取浮点数
            score_match = re.search(r'(\d+\.\d+|\d+)', score_text)
            if score_match:
                consistency_score = float(score_match.group(1))
                # 确保分数在0-1范围内
                return max(0.0, min(1.0, consistency_score))
            return 0.5  # 默认中等一致性
        except Exception as e:
            print(f"使用LLM评估事实一致性时出错: {e}")
            # 回退到简单方法
            return self._assess_consistency_simple(answer, entities, relationships)
    
    def calculate_metric(self, data: RetrievalEvaluationData) -> Tuple[Dict[str, float], List[float]]:
        """
        计算事实一致性评估指标
        
        Args:
            data (RetrievalEvaluationData): 评估数据
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 总体得分和每个样本的得分
        """
        answers = data.system_answers
        retrieved_entities = data.retrieved_entities
        retrieved_relationships = data.retrieved_relationships
        
        consistency_scores = []
        
        # 使用LLM或简单方法评估一致性
        for answer, entities, relationships in zip(answers, retrieved_entities, retrieved_relationships):
            if not answer:
                consistency_scores.append(0.0)
                continue
                
            if self.use_llm:
                consistency = self._assess_consistency_llm(answer, entities, relationships)
            else:
                consistency = self._assess_consistency_simple(answer, entities, relationships)
            
            consistency_scores.append(consistency)
        
        avg_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0
        
        return {"factual_consistency": avg_consistency}, consistency_scores


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
            # 尝试从检索日志中获取延迟信息
            logs = sample.retrieval_logs
            latency = 0.0
            
            if logs:
                # 尝试获取检索时间
                if 'retrieval_time' in logs:
                    latency = logs['retrieval_time']
                    
                # 如果有详细的执行日志，尝试计算检索相关节点的时间
                elif 'execution_log' in logs:
                    execution_log = logs['execution_log']
                    retrieval_nodes = [
                        node for node in execution_log 
                        if node.get('node', '').lower() in ['retrieve', 'retrieval', 'search']
                    ]
                    
                    if retrieval_nodes:
                        # 计算检索节点的执行时间
                        for node in retrieval_nodes:
                            if 'duration' in node:
                                latency += node['duration']
            
            latency_scores.append(latency)
        
        # 计算平均延迟
        avg_latency = sum(latency_scores) / len(latency_scores) if latency_scores else 0.0
        
        return {"retrieval_latency": avg_latency}, latency_scores


class KeywordCoverage(BaseMetric):
    """关键词覆盖率评估指标"""
    
    metric_name = "keyword_coverage"
    
    def __init__(self, config):
        super().__init__(config)
    
    def calculate_metric(self, data: RetrievalEvaluationData) -> Tuple[Dict[str, float], List[float]]:
        """
        计算关键词覆盖率
        
        Args:
            data (RetrievalEvaluationData): 评估数据
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 总体得分和每个样本的得分
        """
        questions = data.questions
        retrieved_entities = data.retrieved_entities
        
        coverage_scores = []
        
        for question, entities in zip(questions, retrieved_entities):
            # 提取问题中的关键词
            keywords = re.findall(r'\b[\w\u4e00-\u9fa5]{2,}\b', normalize_answer(question))
            keywords = [k for k in keywords if len(k) > 1]
            
            if not keywords:
                # 没有关键词，跳过
                coverage_scores.append(0.0)
                continue
                
            # 将检索到的实体名称连接为一个文本
            entities_text = " ".join([normalize_answer(e) for e in entities])
            
            # 计算关键词覆盖率
            covered = sum(1 for k in keywords if k in entities_text)
            coverage = covered / len(keywords) if keywords else 0.0
            
            coverage_scores.append(coverage)
        
        avg_coverage = sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0.0
        
        return {"keyword_coverage": avg_coverage}, coverage_scores


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
                print(f'评估 {metric_name} 时出错!')
                print(e)
                continue
        
        # 保存评估结果
        if self.save_metric_flag:
            self.save_metric_score(result_dict)
        
        # 保存评估数据
        if self.save_data_flag:
            self.save_data(data)
        
        return result_dict
    
    def evaluate_retrieval(self, questions: List[str], agent_with_trace=False) -> Dict[str, float]:
        """
        评估检索系统
        
        Args:
            questions (List[str]): 问题列表
            agent_with_trace (bool): 是否使用带trace的代理
            
        Returns:
            Dict[str, float]: 评估结果
        """
        if not self.qa_agent:
            raise ValueError("未提供QA代理，无法评估检索系统")
        
        # 初始化评估数据
        eval_data = RetrievalEvaluationData()
        
        # 处理每个问题
        for question in questions:
            # 创建评估样本
            sample = RetrievalEvaluationSample(question=question)
            
            # 获取带trace的回答（包含检索数据）
            try:
                if agent_with_trace:
                    response = self.qa_agent.ask_with_trace(question)
                    
                    # 从trace中提取检索数据
                    answer = response.get('answer', '')
                    logs = response.get('execution_log', [])
                    
                    # 提取实体和关系
                    entities, relationships = self._extract_retrieval_data_from_logs(logs)
                    
                    # 更新样本
                    sample.update_answer(answer)
                    sample.update_retrieval_data(entities, relationships)
                    sample.update_logs({"execution_log": logs})
                else:
                    # 普通回答
                    answer = self.qa_agent.ask(question)
                    
                    # 使用Neo4j获取相关图数据
                    if self.neo4j_client:
                        entities, relationships = self._get_relevant_graph_data(question)
                        sample.update_retrieval_data(entities, relationships)
                    
                    # 更新样本
                    sample.update_answer(answer)
            except Exception as e:
                print(f"处理问题 '{question}' 时出错: {e}")
                sample.update_answer("")
            
            # 添加到评估数据
            eval_data.append(sample)
        
        # 执行评估
        results = self.evaluate(eval_data)
        
        # 使用表格格式格式化结果
        formatted_results = self.format_results_table(results)
        print(formatted_results)
        
        return results
    
    def _extract_retrieval_data_from_logs(self, logs: List[Dict[str, Any]]) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """
        从执行日志中提取检索到的实体和关系
        
        Args:
            logs (List[Dict[str, Any]]): 执行日志
            
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
            question (str): 问题
            
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
            MATCH (n)
            WHERE any(word IN $keywords WHERE n.name CONTAINS word OR n.description CONTAINS word)
            RETURN n.name AS name
            LIMIT 10
            """
            
            # 查询与这些实体相关的关系
            relationship_query = """
            MATCH (a)-[r]->(b)
            WHERE a.name IN $entity_names OR b.name IN $entity_names
            RETURN a.name AS source, type(r) AS relation, b.name AS target
            LIMIT 20
            """
            
            # 执行查询
            entity_result = self.neo4j_client.query(entity_query, {"keywords": question_words})
            
            # 提取实体名称
            if entity_result:
                entity_names = [record.get('name', '') for record in entity_result if record.get('name')]
                entities = [name for name in entity_names if name]
                
                # 查询关系
                if entities:
                    rel_result = self.neo4j_client.query(relationship_query, {"entity_names": entities})
                    
                    # 提取关系
                    if rel_result:
                        for record in rel_result:
                            source = record.get('source', '')
                            relation = record.get('relation', '')
                            target = record.get('target', '')
                            if source and relation and target:
                                relationships.append((source, relation, target))
        except Exception as e:
            print(f"从Neo4j获取图数据时出错: {e}")
        
        return entities, relationships