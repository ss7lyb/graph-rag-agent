from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field, asdict
import json

from evaluator.base import BaseEvaluator, BaseMetric
from evaluator.preprocessing import clean_references, clean_thinking_process
from evaluator.utils import normalize_answer, compute_precision_recall_f1

@dataclass
class AnswerEvaluationSample:
    """答案评估样本类，用于存储和更新答案评估数据"""
    
    question: str
    golden_answer: str
    system_answer: str = ""
    scores: Dict[str, float] = field(default_factory=dict)
    agent_type: str = ""  # naive, hybrid, graph, deep
    
    def update_system_answer(self, answer: str, agent_type: str = ""):
        """
        更新系统回答，自动清理引用数据和思考过程
        
        Args:
            answer: 原始系统回答
            agent_type: 代理类型
        """
        # 先清理思考过程，再清理引用数据
        cleaned_answer = clean_thinking_process(answer)
        cleaned_answer = clean_references(cleaned_answer)
        
        self.system_answer = cleaned_answer
        if agent_type:
            self.agent_type = agent_type
    
    def update_evaluation_score(self, metric: str, score: float):
        """更新评估分数"""
        self.scores[metric] = score
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

@dataclass
class AnswerEvaluationData:
    """答案评估数据类，用于管理多个答案评估样本"""
    
    samples: List[AnswerEvaluationSample] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> AnswerEvaluationSample:
        return self.samples[idx]
    
    def append(self, sample: AnswerEvaluationSample):
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
    def system_answers(self) -> List[str]:
        """获取所有系统回答"""
        return [sample.system_answer for sample in self.samples]
    
    def save(self, path: str):
        """保存评估数据"""
        with open(path, "w", encoding='utf-8') as f:
            json.dump([sample.to_dict() for sample in self.samples], f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'AnswerEvaluationData':
        """加载评估数据"""
        with open(path, "r", encoding='utf-8') as f:
            samples_data = json.load(f)
        
        data = cls()
        for sample_data in samples_data:
            sample = AnswerEvaluationSample(**sample_data)
            data.append(sample)
        
        return data

class ExactMatch(BaseMetric):
    """精确匹配评估指标"""
    
    metric_name = "em"
    
    def calculate_em(self, prediction: str, golden_answer: str) -> float:
        """
        计算单个预测的精确匹配得分
        
        Args:
            prediction: 预测答案
            golden_answer: 标准答案
            
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
    
    def calculate_metric(self, data: AnswerEvaluationData) -> Tuple[Dict[str, float], List[float]]:
        """
        计算精确匹配指标
        
        Args:
            data: 评估数据
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 总体得分和每个样本的得分
        """
        golden_answers = data.golden_answers
        system_answers = data.system_answers
        
        metric_score_list = [self.calculate_em(pred, golden) 
                            for pred, golden in zip(system_answers, golden_answers)]
        
        em_score = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0.0
        
        return {"em": em_score}, metric_score_list

class F1Score(BaseMetric):
    """F1分数评估指标"""
    
    metric_name = "f1"
    
    def calculate_metric(self, data: AnswerEvaluationData) -> Tuple[Dict[str, float], List[float]]:
        """
        计算F1分数
        
        Args:
            data: 评估数据
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 总体得分和每个样本的得分
        """
        golden_answers = data.golden_answers
        system_answers = data.system_answers
        
        f1_scores = []
        for pred, golden in zip(system_answers, golden_answers):
            # 将文本分割为单词
            pred_tokens = normalize_answer(pred).split()
            golden_tokens = normalize_answer(golden).split()
            
            # 计算F1分数
            metrics = compute_precision_recall_f1(pred_tokens, golden_tokens)
            f1_scores.append(metrics["f1"])
        
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        
        return {"f1": avg_f1}, f1_scores

class AnswerEvaluator(BaseEvaluator):
    """答案评估器，用于评估系统回答的质量"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化答案评估器
        
        Args:
            config: 评估配置
        """
        super().__init__(config)
    
    def evaluate(self, data: AnswerEvaluationData) -> Dict[str, float]:
        """
        执行评估
        
        Args:
            data: 答案评估数据
            
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