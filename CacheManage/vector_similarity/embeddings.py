import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union
from sentence_transformers import SentenceTransformer
import threading


class EmbeddingProvider(ABC):
    """嵌入向量提供者抽象基类"""
    
    @abstractmethod
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """将文本编码为向量"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """获取向量维度"""
        pass


class SentenceTransformerEmbedding(EmbeddingProvider):
    """基于SentenceTransformer的嵌入向量提供者"""
    
    _instances = {}
    _lock = threading.Lock()
    
    def __new__(cls, model_name: str = 'all-MiniLM-L6-v2'):
        """单例模式，避免重复加载模型"""
        with cls._lock:
            if model_name not in cls._instances:
                cls._instances[model_name] = super().__new__(cls)
                cls._instances[model_name]._initialized = False
            return cls._instances[model_name]
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._dimension = None
        self._initialized = True
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """编码文本为向量"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings
    
    def get_dimension(self) -> int:
        """获取向量维度"""
        if self._dimension is None:
            # 使用一个简单文本获取维度
            test_embedding = self.encode("test")
            self._dimension = test_embedding.shape[-1]
        return self._dimension