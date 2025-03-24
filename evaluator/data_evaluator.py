import os
import json
from typing import Dict, List, Any, Optional, Tuple, Set
import re
from dataclasses import dataclass, field, asdict

from evaluator.base import BaseEvaluator, BaseMetric
from evaluator.utils import (
    normalize_answer, 
    compute_precision_recall_f1, 
    extract_entities_from_neo4j_response,
    extract_relationships_from_neo4j_response
)


@dataclass
class EvaluationSample:
    """评估样本类，用于保存和更新评估数据"""
    
    question: str
    golden_answer: str
    golden_entities: List[str] = field(default_factory=list)
    golden_relationships: List[Tuple[str, str, str]] = field(default_factory=list)
    pred: str = ""
    pred_entities: List[str] = field(default_factory=list)
    pred_relationships: List[Tuple[str, str, str]] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    
    def update_prediction(self, pred: str):
        """更新预测结果"""
        self.pred = pred
        # 提取预测结果中的实体和关系
        self.pred_entities = extract_entities_from_neo4j_response(pred)
        self.pred_relationships = extract_relationships_from_neo4j_response(pred)
    
    def update_evaluation_score(self, metric: str, score: float):
        """更新评估分数"""
        self.scores[metric] = score
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class EvaluationData:
    """评估数据类，用于管理多个评估样本"""
    
    samples: List[EvaluationSample] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> EvaluationSample:
        return self.samples[idx]
    
    def append(self, sample: EvaluationSample):
        """添加评估样本"""
        self.samples.append(sample)
    
    @property
    def questions(self) -> List[str]:
        """获取所有问题"""
        return [sample.question for sample in self.samples]
    
    @property
    def golden_answers(self) -> List[str]:
        """获取所有标准答案"""
        return [sample.golden_answer for sample in self.samples]
    
    @property
    def pred(self) -> List[str]:
        """获取所有预测答案"""
        return [sample.pred for sample in self.samples]
    
    @property
    def golden_entities(self) -> List[List[str]]:
        """获取所有标准实体"""
        return [sample.golden_entities for sample in self.samples]
    
    @property
    def pred_entities(self) -> List[List[str]]:
        """获取所有预测实体"""
        return [sample.pred_entities for sample in self.samples]
    
    @property
    def golden_relationships(self) -> List[List[Tuple[str, str, str]]]:
        """获取所有标准关系"""
        return [sample.golden_relationships for sample in self.samples]
    
    @property
    def pred_relationships(self) -> List[List[Tuple[str, str, str]]]:
        """获取所有预测关系"""
        return [sample.pred_relationships for sample in self.samples]
    
    def save(self, path: str):
        """保存评估数据"""
        with open(path, "w", encoding='utf-8') as f:
            json.dump([sample.to_dict() for sample in self.samples], f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'EvaluationData':
        """加载评估数据"""
        with open(path, "r", encoding='utf-8') as f:
            samples_data = json.load(f)
        
        data = cls()
        for sample_data in samples_data:
            # 转换关系格式（从列表到元组）
            if "golden_relationships" in sample_data:
                sample_data["golden_relationships"] = [tuple(rel) for rel in sample_data["golden_relationships"]]
            if "pred_relationships" in sample_data:
                sample_data["pred_relationships"] = [tuple(rel) for rel in sample_data["pred_relationships"]]
                
            sample = EvaluationSample(**sample_data)
            data.append(sample)
        
        return data


class ExactMatch(BaseMetric):
    """精确匹配评估指标"""
    
    metric_name = "em"
    
    def __init__(self, config):
        super().__init__(config)
    
    def calculate_em(self, prediction: str, golden_answer: str) -> float:
        """
        计算单个预测的精确匹配得分
        
        Args:
            prediction (str): 预测答案
            golden_answer (str): 标准答案
            
        Returns:
            float: 得分（1.0表示匹配，0.0表示不匹配）
        """
        if not prediction or not golden_answer:
            return 0.0
            
        normalized_prediction = normalize_answer(prediction)
        normalized_golden = normalize_answer(golden_answer)
        
        # 完全匹配
        if normalized_prediction == normalized_golden:
            return 1.0
        return 0.0
    
    def calculate_metric(self, data: EvaluationData) -> Tuple[Dict[str, float], List[float]]:
        """
        计算精确匹配指标
        
        Args:
            data (EvaluationData): 评估数据
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 总体得分和每个样本的得分
        """
        golden_answers = data.golden_answers
        pred_list = data.pred
        
        metric_score_list = [self.calculate_em(pred, golden_answer) 
                            for pred, golden_answer in zip(pred_list, golden_answers)]
        
        em_score = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0.0
        
        return {"em": em_score}, metric_score_list


class EntityRecall(BaseMetric):
    """实体召回率评估指标"""
    
    metric_name = "entity_recall"
    
    def __init__(self, config):
        super().__init__(config)
    
    def calculate_metric(self, data: EvaluationData) -> Tuple[Dict[str, float], List[float]]:
        """
        计算实体召回率
        
        Args:
            data (EvaluationData): 评估数据
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 总体得分和每个样本的得分
        """
        golden_entities_list = data.golden_entities
        pred_entities_list = data.pred_entities
        
        recall_scores = []
        for golden_entities, pred_entities in zip(golden_entities_list, pred_entities_list):
            if not golden_entities:
                # 没有标准实体，跳过
                recall_scores.append(1.0)
                continue
            
            # 标准化处理
            golden_norm = [normalize_answer(e) for e in golden_entities]
            pred_norm = [normalize_answer(e) for e in pred_entities]
            
            # 计算召回率
            recalled = sum(1 for e in golden_norm if any(normalize_answer(p) == e for p in pred_norm))
            recall = recalled / len(golden_norm) if golden_norm else 1.0
            recall_scores.append(recall)
        
        avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
        
        return {"entity_recall": avg_recall}, recall_scores


class EntityPrecision(BaseMetric):
    """实体精确率评估指标"""
    
    metric_name = "entity_precision"
    
    def __init__(self, config):
        super().__init__(config)
    
    def calculate_metric(self, data: EvaluationData) -> Tuple[Dict[str, float], List[float]]:
        """
        计算实体精确率
        
        Args:
            data (EvaluationData): 评估数据
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 总体得分和每个样本的得分
        """
        golden_entities_list = data.golden_entities
        pred_entities_list = data.pred_entities
        
        precision_scores = []
        for golden_entities, pred_entities in zip(golden_entities_list, pred_entities_list):
            if not pred_entities:
                # 没有预测实体，精确率为0
                precision_scores.append(0.0)
                continue
            
            # 标准化处理
            golden_norm = [normalize_answer(e) for e in golden_entities]
            pred_norm = [normalize_answer(e) for e in pred_entities]
            
            # 计算精确率
            correct = sum(1 for p in pred_norm if any(normalize_answer(g) == p for g in golden_norm))
            precision = correct / len(pred_norm) if pred_norm else 0.0
            precision_scores.append(precision)
        
        avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
        
        return {"entity_precision": avg_precision}, precision_scores


class EntityF1(BaseMetric):
    """实体F1评估指标"""
    
    metric_name = "entity_f1"
    
    def __init__(self, config):
        super().__init__(config)
    
    def calculate_metric(self, data: EvaluationData) -> Tuple[Dict[str, float], List[float]]:
        """
        计算实体F1分数
        
        Args:
            data (EvaluationData): 评估数据
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 总体得分和每个样本的得分
        """
        golden_entities_list = data.golden_entities
        pred_entities_list = data.pred_entities
        
        f1_scores = []
        for golden_entities, pred_entities in zip(golden_entities_list, pred_entities_list):
            # 计算F1分数
            metrics = compute_precision_recall_f1(pred_entities, golden_entities)
            f1_scores.append(metrics["f1"])
        
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        
        return {"entity_f1": avg_f1}, f1_scores


class RelationshipRecall(BaseMetric):
    """关系召回率评估指标"""
    
    metric_name = "relationship_recall"
    
    def __init__(self, config):
        super().__init__(config)
    
    def calculate_metric(self, data: EvaluationData) -> Tuple[Dict[str, float], List[float]]:
        """
        计算关系召回率
        
        Args:
            data (EvaluationData): 评估数据
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 总体得分和每个样本的得分
        """
        golden_rels_list = data.golden_relationships
        pred_rels_list = data.pred_relationships
        
        recall_scores = []
        for golden_rels, pred_rels in zip(golden_rels_list, pred_rels_list):
            if not golden_rels:
                # 没有标准关系，跳过
                recall_scores.append(1.0)
                continue
            
            # 标准化处理
            golden_norm = [(normalize_answer(src), normalize_answer(rel), normalize_answer(dst)) 
                          for src, rel, dst in golden_rels]
            pred_norm = [(normalize_answer(src), normalize_answer(rel), normalize_answer(dst)) 
                        for src, rel, dst in pred_rels]
            
            # 计算召回率
            recalled = 0
            for g_src, g_rel, g_dst in golden_norm:
                for p_src, p_rel, p_dst in pred_norm:
                    # 源实体和目标实体都匹配，关系类型也匹配
                    if g_src == p_src and g_dst == p_dst and g_rel == p_rel:
                        recalled += 1
                        break
            
            recall = recalled / len(golden_norm) if golden_norm else 1.0
            recall_scores.append(recall)


class RelationshipRecall(BaseMetric):
    """关系召回率评估指标"""
    
    metric_name = "relationship_recall"
    
    def __init__(self, config):
        super().__init__(config)
    
    def calculate_metric(self, data: EvaluationData) -> Tuple[Dict[str, float], List[float]]:
        """
        计算关系召回率
        
        Args:
            data (EvaluationData): 评估数据
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 总体得分和每个样本的得分
        """
        golden_rels_list = data.golden_relationships
        pred_rels_list = data.pred_relationships
        
        recall_scores = []
        for golden_rels, pred_rels in zip(golden_rels_list, pred_rels_list):
            if not golden_rels:
                # 没有标准关系，跳过
                recall_scores.append(1.0)
                continue
            
            # 标准化处理
            golden_norm = [(normalize_answer(src), normalize_answer(rel), normalize_answer(dst)) 
                          for src, rel, dst in golden_rels]
            pred_norm = [(normalize_answer(src), normalize_answer(rel), normalize_answer(dst)) 
                        for src, rel, dst in pred_rels]
            
            # 计算召回率
            recalled = 0
            for g_src, g_rel, g_dst in golden_norm:
                for p_src, p_rel, p_dst in pred_norm:
                    # 源实体和目标实体都匹配，关系类型也匹配
                    if g_src == p_src and g_dst == p_dst and g_rel == p_rel:
                        recalled += 1
                        break
            
            recall = recalled / len(golden_norm) if golden_norm else 1.0
            recall_scores.append(recall)