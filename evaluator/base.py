import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseMetric(ABC):
    """所有评估指标的基类"""
    
    metric_name = "base"
    
    def __init__(self, config):
        """
        初始化评估指标基类
        
        Args:
            config (Dict): 评估配置
        """
        self.config = config
        self.dataset_name = config.get('dataset_name', 'default')
    
    @abstractmethod
    def calculate_metric(self, data):
        """
        计算评估指标
        
        Args:
            data: 评估数据对象
            
        Returns:
            Tuple[Dict, List]: 评估结果和每个样本的评分
        """
        return {}, []


class BaseEvaluator(ABC):
    """评估器基类，定义通用评估功能和接口"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化评估器
        
        Args:
            config (Dict): 评估配置
        """
        self.config = config
        self.save_dir = config.get('save_dir', './evaluation_results')
        self.save_metric_flag = config.get('save_metric_score', True)
        self.save_data_flag = config.get('save_intermediate_data', True)
        self.metrics = [metric.lower() for metric in config.get('metrics', [])]
        
        # 确保保存目录存在
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 获取可用的评估指标
        self.available_metrics = self._collect_metrics()
        
        # 初始化评估指标
        self.metric_class = {}
        for metric in self.metrics:
            if metric in self.available_metrics:
                self.metric_class[metric] = self.available_metrics[metric](self.config)
            else:
                print(f"{metric} 评估指标未实现!")
                raise NotImplementedError
    
    def _collect_metrics(self):
        """收集所有继承自BaseMetric的评估指标类"""
        
        def find_descendants(base_class, subclasses=None):
            if subclasses is None:
                subclasses = set()
            
            direct_subclasses = base_class.__subclasses__()
            for subclass in direct_subclasses:
                if subclass not in subclasses:
                    subclasses.add(subclass)
                    find_descendants(subclass, subclasses)
            return subclasses
        
        available_metrics = {}
        for cls in find_descendants(BaseMetric):
            metric_name = cls.metric_name
            available_metrics[metric_name] = cls
        return available_metrics
    
    @abstractmethod
    def evaluate(self, data):
        """
        执行评估
        
        Args:
            data: 评估数据
            
        Returns:
            Dict: 评估结果
        """
        pass
    
    def save_metric_score(self, result_dict: Dict[str, float]):
        """
        保存评估指标结果
        
        Args:
            result_dict (Dict): 评估结果字典
        """
        file_name = "metric_score.txt"
        save_path = os.path.join(self.save_dir, file_name)
        with open(save_path, "w", encoding='utf-8') as f:
            for k, v in result_dict.items():
                f.write(f"{k}: {v}\n")
    
    def save_data(self, data):
        """
        保存评估中间数据
        
        Args:
            data: 评估数据对象
        """
        file_name = "intermediate_data.json"
        save_path = os.path.join(self.save_dir, file_name)
        
        # 检查data对象是否有save方法
        if hasattr(data, 'save'):
            data.save(save_path)
        else:
            # 如果没有save方法，尝试将其转换为可序列化的字典
            try:
                serializable_data = self._convert_to_serializable(data)
                with open(save_path, "w", encoding='utf-8') as f:
                    json.dump(serializable_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"保存数据时出错: {e}")
    
    def _convert_to_serializable(self, data):
        """
        将数据转换为可序列化的格式
        
        Args:
            data: 要转换的数据
            
        Returns:
            Dict: 可序列化的字典
        """
        if isinstance(data, dict):
            return {k: self._convert_to_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_to_serializable(item) for item in data]
        elif hasattr(data, '__dict__'):
            return self._convert_to_serializable(data.__dict__)
        else:
            return data
    
    def format_results_table(self, results: Dict[str, float]) -> str:
        """
        将评估结果格式化为表格形式
        
        Args:
            results (Dict): 评估结果字典
            
        Returns:
            str: 格式化的表格字符串
        """
        header = "| 指标 | 得分 |"
        separator = "| --- | --- |"
        
        rows = []
        for metric, score in results.items():
            # 如果得分是浮点数，保留4位小数
            if isinstance(score, float):
                score_str = f"{score:.4f}"
            else:
                score_str = str(score)
            rows.append(f"| {metric} | {score_str} |")
        
        table = "\n".join([header, separator] + rows)
        return table