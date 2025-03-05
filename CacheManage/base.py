import os
import json
import time
import hashlib
from typing import Any, Dict, Optional, Callable
from abc import ABC, abstractmethod


class CacheKeyStrategy(ABC):
    """缓存键生成策略的抽象基类"""
    
    @abstractmethod
    def generate_key(self, query: str, **kwargs) -> str:
        """生成缓存键"""
        pass


class SimpleCacheKeyStrategy(CacheKeyStrategy):
    """简单的MD5哈希缓存键策略"""
    
    def generate_key(self, query: str, **kwargs) -> str:
        """使用查询字符串的MD5哈希生成缓存键"""
        return hashlib.md5(query.strip().encode('utf-8')).hexdigest()


class ContextAwareCacheKeyStrategy(CacheKeyStrategy):
    """上下文感知的缓存键策略，考虑会话历史"""
    
    def __init__(self, context_window: int = 3):
        """
        初始化上下文感知缓存键策略
        
        参数:
            context_window: 要考虑的前几条会话历史记录
        """
        self.context_window = context_window
        self.conversation_history = {}
    
    def update_history(self, query: str, thread_id: str = "default", max_history: int = 10):
        """更新会话历史"""
        if thread_id not in self.conversation_history:
            self.conversation_history[thread_id] = []
        
        # 添加新查询到历史
        self.conversation_history[thread_id].append(query)
        
        # 保持历史记录在可管理的大小
        if len(self.conversation_history[thread_id]) > max_history:
            self.conversation_history[thread_id] = self.conversation_history[thread_id][-max_history:]
    
    def generate_key(self, query: str, **kwargs) -> str:
        """
        生成上下文感知的缓存键
        
        参数:
            query: 查询字符串
            thread_id: 会话ID，默认为"default"
        
        返回:
            str: 缓存键
        """
        thread_id = kwargs.get("thread_id", "default")
        
        # 获取当前会话的历史记录
        history = self.conversation_history.get(thread_id, [])
        
        # 构建上下文字符串 - 包含最近的n条消息
        context = " ".join(history[-self.context_window:] if self.context_window > 0 else [])
        
        # 组合上下文和查询生成缓存键
        combined = (context + " " + query).strip()
        return hashlib.md5(combined.encode('utf-8')).hexdigest()


class ContextAndKeywordAwareCacheKeyStrategy(CacheKeyStrategy):
    """结合上下文和关键词的缓存键策略，同时考虑会话历史和关键词"""
    
    def __init__(self, context_window: int = 3):
        """
        初始化上下文与关键词感知的缓存键策略
        
        参数:
            context_window: 要考虑的前几条会话历史记录
        """
        self.context_window = context_window
        self.conversation_history = {}
    
    def update_history(self, query: str, thread_id: str = "default", max_history: int = 10):
        """更新会话历史"""
        if thread_id not in self.conversation_history:
            self.conversation_history[thread_id] = []
        
        # 添加新查询到历史
        self.conversation_history[thread_id].append(query)
        
        # 保持历史记录在可管理的大小
        if len(self.conversation_history[thread_id]) > max_history:
            self.conversation_history[thread_id] = self.conversation_history[thread_id][-max_history:]
    
    def generate_key(self, query: str, **kwargs) -> str:
        """
        生成同时考虑上下文和关键词的缓存键
        
        参数:
            query: 查询字符串
            thread_id: 会话ID，默认为"default"
            low_level_keywords: 低级关键词列表
            high_level_keywords: 高级关键词列表
        
        返回:
            str: 缓存键
        """
        thread_id = kwargs.get("thread_id", "default")
        key_parts = [query.strip()]
        
        # 添加上下文信息
        # 获取当前会话的历史记录
        history = self.conversation_history.get(thread_id, [])
        
        # 构建上下文字符串 - 包含最近的n条消息
        if self.context_window > 0 and history:
            context = " ".join(history[-self.context_window:])
            key_parts.append("ctx:" + hashlib.md5(context.encode('utf-8')).hexdigest())
        
        # 添加低级关键词
        low_level_keywords = kwargs.get("low_level_keywords", [])
        if low_level_keywords:
            key_parts.append("low:" + ",".join(sorted(low_level_keywords)))
        
        # 添加高级关键词
        high_level_keywords = kwargs.get("high_level_keywords", [])
        if high_level_keywords:
            key_parts.append("high:" + ",".join(sorted(high_level_keywords)))
        
        # 生成最终的键
        key_str = "||".join(key_parts)
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()


class CacheStorageBackend(ABC):
    """缓存存储后端的抽象基类"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """设置缓存项"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """删除缓存项"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """清空缓存"""
        pass


class MemoryCacheBackend(CacheStorageBackend):
    """内存缓存后端实现"""
    
    def __init__(self, max_size: int = 100):
        """
        初始化内存缓存后端
        
        参数:
            max_size: 缓存最大项数
        """
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}  # 用于LRU淘汰策略
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存项
        
        参数:
            key: 缓存键
            
        返回:
            Optional[Any]: 缓存项值，不存在则返回None
        """
        value = self.cache.get(key)
        if value is not None:
            # 更新访问时间（LRU策略）
            self.access_times[key] = time.time()
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        设置缓存项
        
        参数:
            key: 缓存键
            value: 缓存值
        """
        # 如果缓存已满，删除最久未使用的项
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict_lru()
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def delete(self, key: str) -> bool:
        """
        删除缓存项
        
        参数:
            key: 缓存键
            
        返回:
            bool: 是否成功删除
        """
        if key in self.cache:
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
            return True
        return False
    
    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
        self.access_times.clear()
    
    def _evict_lru(self) -> None:
        """淘汰最久未使用的缓存项"""
        if not self.access_times:
            return
            
        # 找出最旧的项
        oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        self.delete(oldest_key)


class DiskCacheBackend(CacheStorageBackend):
    """磁盘缓存后端实现"""
    
    def __init__(self, cache_dir: str = "./cache", max_size: int = 1000):
        """
        初始化磁盘缓存后端
        
        参数:
            cache_dir: 缓存目录
            max_size: 缓存最大项数
        """
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.metadata = {}  # 用于存储元数据
        self.write_queue = []  # 写入队列
        self.last_flush_time = time.time()
        
        # 确保缓存目录存在
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # 加载索引
        self._load_index()
    
    def _get_cache_path(self, key: str) -> str:
        """
        获取缓存文件路径
        
        参数:
            key: 缓存键
            
        返回:
            str: 缓存文件路径
        """
        return os.path.join(self.cache_dir, f"{key}.json")
    
    def _get_index_path(self) -> str:
        """
        获取索引文件路径
        
        返回:
            str: 索引文件路径
        """
        return os.path.join(self.cache_dir, "index.json")
    
    def _load_index(self) -> None:
        """加载缓存索引"""
        index_path = self._get_index_path()
        if os.path.exists(index_path):
            try:
                with open(index_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                print(f"加载缓存索引失败: {e}")
                self.metadata = {}
        
        # 验证磁盘上的文件
        files = [f[:-5] for f in os.listdir(self.cache_dir) if f.endswith(".json") and f != "index.json"]
        
        # 同步索引和文件
        for key in list(self.metadata.keys()):
            if key not in files:
                del self.metadata[key]
        
        for key in files:
            if key not in self.metadata:
                self.metadata[key] = {"created_at": time.time(), "access_count": 0}
    
    def _save_index(self) -> None:
        """保存缓存索引"""
        try:
            with open(self._get_index_path(), 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存缓存索引失败: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存项
        
        参数:
            key: 缓存键
            
        返回:
            Optional[Any]: 缓存项值，不存在则返回None
        """
        cache_path = self._get_cache_path(key)
        if key in self.metadata and os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    value = json.load(f)
                
                # 更新访问信息
                self.metadata[key]["last_accessed"] = time.time()
                self.metadata[key]["access_count"] = self.metadata[key].get("access_count", 0) + 1
                
                # 异步保存索引（非阻塞）
                self._save_index_async()
                
                return value
            except Exception as e:
                print(f"读取缓存文件失败: {e}")
        
        return None
    
    def set(self, key: str, value: Any) -> None:
        """
        设置缓存项
        
        参数:
            key: 缓存键
            value: 缓存值
        """
        # 如果缓存已满，淘汰项
        if len(self.metadata) >= self.max_size and key not in self.metadata:
            self._evict_items()
        
        # 更新元数据
        self.metadata[key] = {
            "created_at": time.time(),
            "access_count": 0,
            "last_accessed": time.time()
        }
        
        # 添加到写入队列
        self.write_queue.append((key, value))
        
        # 根据条件决定是否立即刷新
        current_time = time.time()
        if len(self.write_queue) >= 10 or (current_time - self.last_flush_time) > 30:
            self._flush_write_queue()
    
    def _flush_write_queue(self) -> None:
        """刷新写入队列"""
        if not self.write_queue:
            return
        
        for key, value in self.write_queue:
            try:
                with open(self._get_cache_path(key), 'w', encoding='utf-8') as f:
                    json.dump(value, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"写入缓存文件失败: {e}")
        
        # 清空队列并更新时间戳
        self.write_queue = []
        self.last_flush_time = time.time()
        
        # 保存索引
        self._save_index()
    
    def _save_index_async(self) -> None:
        """异步保存索引（简化实现）"""
        # 在真实环境中，应该使用异步方法执行
        # 这里简化为定期保存
        current_time = time.time()
        if current_time - self.last_flush_time > 60:  # 每分钟最多保存一次
            self._save_index()
            self.last_flush_time = current_time
    
    def delete(self, key: str) -> bool:
        """
        删除缓存项
        
        参数:
            key: 缓存键
            
        返回:
            bool: 是否成功删除
        """
        if key in self.metadata:
            # 从元数据中删除
            del self.metadata[key]
            
            # 删除文件
            cache_path = self._get_cache_path(key)
            if os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                except Exception as e:
                    print(f"删除缓存文件失败: {e}")
                    return False
            
            # 保存索引
            self._save_index()
            return True
        
        return False
    
    def clear(self) -> None:
        """清空缓存"""
        # 清空写入队列
        self.write_queue = []
        
        # 清空元数据
        self.metadata = {}
        
        # 删除所有缓存文件
        for filename in os.listdir(self.cache_dir):
            if filename.endswith(".json"):
                try:
                    os.remove(os.path.join(self.cache_dir, filename))
                except Exception as e:
                    print(f"删除缓存文件失败: {e}")
        
        # 保存空索引
        self._save_index()
    
    def _evict_items(self) -> None:
        """淘汰缓存项，使用复合策略：优先淘汰访问量低且旧的项"""
        if not self.metadata:
            return
        
        # 计算每个项的分数：score = 访问次数 / (当前时间 - 创建时间)
        scores = {}
        current_time = time.time()
        
        for key, meta in self.metadata.items():
            age = current_time - meta.get("created_at", current_time)
            access_count = meta.get("access_count", 0)
            
            # 避免除以零
            if age < 1:
                age = 1
            
            # 分数越高表示越应该保留
            scores[key] = access_count / age
        
        # 找出分数最低的前10%项
        num_to_evict = max(1, len(scores) // 10)
        keys_to_evict = sorted(scores.keys(), key=lambda k: scores[k])[:num_to_evict]
        
        # 删除这些项
        for key in keys_to_evict:
            self.delete(key)


class HybridCacheBackend(CacheStorageBackend):
    """混合缓存后端实现（内存+磁盘）"""
    
    def __init__(self, cache_dir: str = "./cache", memory_max_size: int = 100, disk_max_size: int = 1000):
        """
        初始化混合缓存后端
        
        参数:
            cache_dir: 缓存目录
            memory_max_size: 内存缓存最大项数
            disk_max_size: 磁盘缓存最大项数
        """
        self.memory_cache = MemoryCacheBackend(max_size=memory_max_size)
        self.disk_cache = DiskCacheBackend(cache_dir=cache_dir, max_size=disk_max_size)
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存项，先检查内存再检查磁盘
        
        参数:
            key: 缓存键
            
        返回:
            Optional[Any]: 缓存项值，不存在则返回None
        """
        # 首先检查内存缓存
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # 如果内存中没有，检查磁盘缓存
        value = self.disk_cache.get(key)
        if value is not None:
            # 将磁盘中的项添加到内存缓存
            self.memory_cache.set(key, value)
            return value
        
        return None
    
    def set(self, key: str, value: Any) -> None:
        """
        设置缓存项，同时更新内存和磁盘缓存
        
        参数:
            key: 缓存键
            value: 缓存值
        """
        # 同时更新内存和磁盘缓存
        self.memory_cache.set(key, value)
        self.disk_cache.set(key, value)
    
    def delete(self, key: str) -> bool:
        """
        删除缓存项
        
        参数:
            key: 缓存键
            
        返回:
            bool: 是否成功删除
        """
        memory_success = self.memory_cache.delete(key)
        disk_success = self.disk_cache.delete(key)
        return memory_success or disk_success
    
    def clear(self) -> None:
        """清空缓存"""
        self.memory_cache.clear()
        self.disk_cache.clear()


class CacheItem:
    """缓存项包装类，支持元数据"""
    
    def __init__(self, content: Any, metadata: Optional[Dict[str, Any]] = None):
        """
        初始化缓存项
        
        参数:
            content: 缓存内容
            metadata: 元数据字典
        """
        self.content = content
        self.metadata = metadata or {
            "created_at": time.time(),
            "quality_score": 0,
            "user_verified": False,
            "access_count": 0
        }
    
    def get_content(self) -> Any:
        """
        获取内容
        
        返回:
            Any: 缓存内容
        """
        return self.content
    
    def is_high_quality(self) -> bool:
        """
        判断是否为高质量缓存
        
        返回:
            bool: 是否为高质量缓存
        """
        return self.metadata.get("user_verified", False) or self.metadata.get("quality_score", 0) > 2
    
    def mark_quality(self, is_positive: bool) -> None:
        """
        标记缓存质量
        
        参数:
            is_positive: 是否为正面评价
        """
        if is_positive:
            current_score = self.metadata.get("quality_score", 0)
            self.metadata["quality_score"] = current_score + 1
            self.metadata["user_verified"] = True
        else:
            current_score = self.metadata.get("quality_score", 0)
            self.metadata["quality_score"] = max(0, current_score - 2)  # 负面评价权重更大
    
    def update_access_stats(self) -> None:
        """更新访问统计"""
        self.metadata["access_count"] = self.metadata.get("access_count", 0) + 1
        self.metadata["last_accessed"] = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        返回:
            Dict[str, Any]: 包含内容和元数据的字典
        """
        return {
            "content": self.content,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheItem':
        """
        从字典创建缓存项
        
        参数:
            data: 字典数据
            
        返回:
            CacheItem: 缓存项
        """
        if isinstance(data, dict) and "content" in data and "metadata" in data:
            return cls(data["content"], data["metadata"])
        else:
            # 尝试兼容旧格式
            return cls(data, {
                "created_at": time.time(),
                "quality_score": 0,
                "user_verified": False,
                "access_count": 0
            })
    
    @classmethod
    def from_any(cls, data: Any) -> 'CacheItem':
        """
        从任意数据创建缓存项，具有自动类型检测
        
        参数:
            data: 任意数据
            
        返回:
            CacheItem: 缓存项
        """
        if isinstance(data, cls):
            return data
        elif isinstance(data, dict) and "content" in data:
            return cls.from_dict(data)
        else:
            return cls(data)


class CacheManager:
    """统一缓存管理器，提供高级缓存功能"""
    
    def __init__(self, 
                 key_strategy: CacheKeyStrategy = None, 
                 storage_backend: CacheStorageBackend = None,
                 cache_dir: str = "./cache",
                 memory_only: bool = False,
                 max_memory_size: int = 100,
                 max_disk_size: int = 1000):
        """
        初始化缓存管理器
        
        参数:
            key_strategy: 缓存键生成策略
            storage_backend: 存储后端
            cache_dir: 缓存目录
            memory_only: 是否仅使用内存缓存
            max_memory_size: 内存缓存最大大小
            max_disk_size: 磁盘缓存最大大小
        """
        # 设置缓存键策略
        self.key_strategy = key_strategy or SimpleCacheKeyStrategy()
        
        # 设置存储后端
        if storage_backend:
            self.storage = storage_backend
        elif memory_only:
            self.storage = MemoryCacheBackend(max_size=max_memory_size)
        else:
            self.storage = HybridCacheBackend(
                cache_dir=cache_dir,
                memory_max_size=max_memory_size,
                disk_max_size=max_disk_size
            )
        
        # 性能指标收集
        self.performance_metrics = {}
    
    def get(self, query: str, skip_validation: bool = False, **kwargs) -> Optional[Any]:
        """
        获取缓存内容
        
        参数:
            query: 查询字符串
            skip_validation: 是否跳过验证
            **kwargs: 其他参数，传递给缓存键策略
        
        返回:
            Optional[Any]: 缓存内容，不存在则返回None
        """
        start_time = time.time()
        
        # 生成缓存键
        key = self.key_strategy.generate_key(query, **kwargs)
        
        # 获取缓存项
        cached_data = self.storage.get(key)
        if cached_data is None:
            self.performance_metrics["get_time"] = time.time() - start_time
            return None
        
        # 包装为缓存项
        cache_item = CacheItem.from_any(cached_data)
        
        # 更新访问统计
        cache_item.update_access_stats()
        
        # 如果是高质量缓存或跳过验证，直接返回内容
        if skip_validation or cache_item.is_high_quality():
            content = cache_item.get_content()
            self.performance_metrics["get_time"] = time.time() - start_time
            return content
        
        # 返回缓存内容
        content = cache_item.get_content()
        self.performance_metrics["get_time"] = time.time() - start_time
        return content
    
    def get_fast(self, query: str, **kwargs) -> Optional[Any]:
        """
        快速获取高质量缓存内容
        
        参数:
            query: 查询字符串
            **kwargs: 其他参数，传递给缓存键策略
        
        返回:
            Optional[Any]: 缓存内容，非高质量缓存则返回None
        """
        start_time = time.time()
        
        # 生成缓存键
        key = self.key_strategy.generate_key(query, **kwargs)
        
        # 获取缓存项
        cached_data = self.storage.get(key)
        if cached_data is None:
            self.performance_metrics["fast_get_time"] = time.time() - start_time
            return None
        
        # 包装为缓存项
        cache_item = CacheItem.from_any(cached_data)
        
        # 只返回高质量缓存
        if cache_item.is_high_quality():
            # 更新访问统计
            cache_item.update_access_stats()
            
            # 如果是上下文感知的策略，更新历史
            if isinstance(self.key_strategy, ContextAwareCacheKeyStrategy) or \
            isinstance(self.key_strategy, ContextAndKeywordAwareCacheKeyStrategy):
                thread_id = kwargs.get("thread_id", "default")
                self.key_strategy.update_history(query, thread_id)
            
            content = cache_item.get_content()
            self.performance_metrics["fast_get_time"] = time.time() - start_time
            return content
        
        self.performance_metrics["fast_get_time"] = time.time() - start_time
        return None
    
    def set(self, query: str, result: Any, **kwargs) -> None:
        """
        设置缓存内容
        
        参数:
            query: 查询字符串
            result: 缓存内容
            **kwargs: 其他参数，传递给缓存键策略
        """
        start_time = time.time()
        
        # 如果key_strategy是上下文感知的，更新会话历史
        if isinstance(self.key_strategy, ContextAwareCacheKeyStrategy):
            thread_id = kwargs.get("thread_id", "default")
            self.key_strategy.update_history(query, thread_id)
        
        # 生成缓存键
        key = self.key_strategy.generate_key(query, **kwargs)
        
        # 包装缓存项
        if isinstance(result, dict) and "content" in result and "metadata" in result:
            # 已经是缓存项格式
            cache_item = CacheItem.from_dict(result)
        else:
            # 创建新的缓存项
            cache_item = CacheItem(result)
        
        # 存储缓存项
        self.storage.set(key, cache_item.to_dict())
        
        self.performance_metrics["set_time"] = time.time() - start_time
    
    def mark_quality(self, query: str, is_positive: bool, **kwargs) -> bool:
        """
        标记缓存质量
        
        参数:
            query: 查询字符串
            is_positive: 是否为正面评价
            **kwargs: 其他参数，传递给缓存键策略
        
        返回:
            bool: 是否成功标记
        """
        start_time = time.time()
        
        # 生成缓存键
        key = self.key_strategy.generate_key(query, **kwargs)
        
        # 获取缓存项
        cached_data = self.storage.get(key)
        if cached_data is None:
            self.performance_metrics["mark_time"] = time.time() - start_time
            return False
        
        # 包装为缓存项
        cache_item = CacheItem.from_any(cached_data)
        
        # 标记质量
        cache_item.mark_quality(is_positive)
        
        # 更新缓存
        self.storage.set(key, cache_item.to_dict())
        
        self.performance_metrics["mark_time"] = time.time() - start_time
        return True
    
    def delete(self, query: str, **kwargs) -> bool:
        """
        删除缓存项
        
        参数:
            query: 查询字符串
            **kwargs: 其他参数，传递给缓存键策略
        
        返回:
            bool: 是否成功删除
        """
        # 生成缓存键
        key = self.key_strategy.generate_key(query, **kwargs)
        
        # 删除缓存项
        return self.storage.delete(key)
    
    def clear(self) -> None:
        """清空缓存"""
        self.storage.clear()
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        获取性能指标
        
        返回:
            Dict[str, Any]: 性能指标字典
        """
        return self.performance_metrics
    
    def validate_answer(self, query: str, answer: str, validator: Callable[[str, str], bool] = None, **kwargs) -> bool:
        """
        验证答案质量
        
        参数:
            query: 查询字符串
            answer: 答案内容
            validator: 自定义验证函数
            **kwargs: 其他参数，传递给缓存键策略
        
        返回:
            bool: 答案是否有效
        """
        # 生成缓存键
        key = self.key_strategy.generate_key(query, **kwargs)
        
        # 获取缓存项
        cached_data = self.storage.get(key)
        if cached_data is None:
            # 如果缓存不存在，直接使用验证函数
            if validator:
                return validator(query, answer)
            return True  # 默认为有效
        
        # 包装为缓存项
        cache_item = CacheItem.from_any(cached_data)
        
        # 检查用户验证状态
        if cache_item.metadata.get("user_verified", False):
            return True  # 用户已验证的答案直接视为有效
        
        # 检查质量分数
        quality_score = cache_item.metadata.get("quality_score", 0)
        if quality_score < 0:
            return False  # 负面评价的答案视为无效
        
        # 如果提供了自定义验证函数，使用它
        if validator:
            return validator(query, answer)
        
        # 基本验证：长度检查
        if len(answer) < 20:
            return False
        
        return True  # 默认为有效