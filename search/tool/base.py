from abc import ABC, abstractmethod
from typing import List, Dict, Any
import time

from langchain_core.tools import BaseTool

from model.get_models import get_llm_model, get_embeddings_model
from CacheManage.base import CacheManager, ContextAndKeywordAwareCacheKeyStrategy, MemoryCacheBackend
from config.neo4jdb import get_db_manager


class BaseSearchTool(ABC):
    """搜索工具基础类"""
    
    def __init__(self, cache_dir: str = "./cache/search"):
        """
        初始化搜索工具
        
        参数:
            cache_dir: 缓存目录
        """
        # 初始化模型
        self.llm = get_llm_model()
        self.embeddings = get_embeddings_model()
        
        # 初始化缓存管理器
        self.cache_manager = CacheManager(
            key_strategy=ContextAndKeywordAwareCacheKeyStrategy(),
            storage_backend=MemoryCacheBackend(max_size=200),
            cache_dir=cache_dir
        )
        
        # 性能监控
        self.performance_metrics = {
            "query_time": 0,
            "llm_time": 0,
            "total_time": 0
        }
        
        # 初始化Neo4j连接
        self._setup_neo4j()
    
    def _setup_neo4j(self):
        """设置Neo4j连接"""
        # 获取连接管理器
        db_manager = get_db_manager()
        
        # 获取图实例
        self.graph = db_manager.get_graph()
        
        # 获取驱动（如果需要直接执行查询）
        self.driver = db_manager.get_driver()
    
    def db_query(self, cypher: str, params: Dict[str, Any] = {}):
        """执行Cypher查询"""
        # 使用连接管理器执行查询
        return get_db_manager().execute_query(cypher, params)
        
    @abstractmethod
    def _setup_chains(self):
        """设置处理链，子类必须实现"""
        pass
    
    @abstractmethod
    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        从查询中提取关键词
        
        参数:
            query: 查询字符串
            
        返回:
            Dict[str, List[str]]: 关键词字典，包含低级和高级关键词
        """
        pass
    
    @abstractmethod
    def search(self, query: Any) -> str:
        """
        执行搜索
        
        参数:
            query: 查询内容，可以是字符串或包含更多信息的字典
            
        返回:
            str: 搜索结果
        """
        pass
    
    def get_tool(self) -> BaseTool:
        """
        获取搜索工具
        
        返回:
            BaseTool: 搜索工具
        """
        # 创建动态工具类
        class DynamicSearchTool(BaseTool):
            name = f"{self.__class__.__name__.lower()}"
            description = "高级搜索工具，用于在知识库中查找信息"
            
            def _run(self_tool, query: Any) -> str:
                return self.search(query)
            
            def _arun(self_tool, query: Any) -> str:
                raise NotImplementedError("异步执行未实现")
        
        return DynamicSearchTool()
    
    def _log_performance(self, operation: str, start_time: float):
        """
        记录性能指标
        
        参数:
            operation: 操作名称
            start_time: 开始时间
        """
        duration = time.time() - start_time
        self.performance_metrics[operation] = duration
        print(f"性能指标 - {operation}: {duration:.4f}s")
    
    def close(self):
        """关闭资源"""
        # 关闭Neo4j连接
        if hasattr(self, 'graph'):
            # 如果Neo4jGraph有close方法，调用它
            if hasattr(self.graph, 'close'):
                self.graph.close()
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()