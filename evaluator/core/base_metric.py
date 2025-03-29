from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

class BaseMetric(ABC):
    """所有评估指标的基类"""
    
    # 指标名称，子类必须重写
    metric_name = "base"
    
    def __init__(self, config):
        """
        初始化评估指标基类
        
        Args:
            config: 评估配置
        """
        # 支持字典或EvaluatorConfig对象
        if isinstance(config, dict):
            from evaluator.evaluator_config.evaluatorConfig import EvaluatorConfig
            self.config = EvaluatorConfig(config)
        else:
            self.config = config
            
        self.dataset_name = self.config.get('dataset_name', 'default')
        self.debug = self.config.get('debug', False)
    
    @abstractmethod
    def calculate_metric(self, data) -> Tuple[Dict[str, float], List]:
        """
        计算评估指标
        
        Args:
            data: 评估数据对象
            
        Returns:
            Tuple[Dict, List]: 评估结果和每个样本的评分
        """
        return {}, []
    
    def log(self, message, *args, **kwargs):
        """
        输出调试日志
        
        Args:
            message: 日志消息
            *args, **kwargs: 额外参数
        """
        from evaluator import debug_print
        if self.debug:
            debug_print(f"[{self.__class__.__name__}] {message}", *args, **kwargs)