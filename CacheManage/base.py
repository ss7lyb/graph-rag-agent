import os
import json
import time
import hashlib
from typing import Any, Dict, Optional, Callable
from abc import ABC, abstractmethod
import threading


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
        self.history_versions = {}
    
    def update_history(self, query: str, thread_id: str = "default", max_history: int = 10):
        """更新会话历史"""
        if thread_id not in self.conversation_history:
            self.conversation_history[thread_id] = []
            self.history_versions[thread_id] = 0  # 初始化版本号
        
        # 添加新查询到历史
        self.conversation_history[thread_id].append(query)
        
        # 保持历史记录在可管理的大小
        if len(self.conversation_history[thread_id]) > max_history:
            self.conversation_history[thread_id] = self.conversation_history[thread_id][-max_history:]
        
        # 增加版本号，确保上下文变化时键也会变化
        self.history_versions[thread_id] += 1
    
    def generate_key(self, query: str, **kwargs) -> str:
        """生成上下文感知的缓存键"""
        thread_id = kwargs.get("thread_id", "default")
        
        # 获取当前会话的历史记录
        history = self.conversation_history.get(thread_id, [])
        
        # 获取历史版本号
        version = self.history_versions.get(thread_id, 0)
        
        # 构建上下文字符串 - 包含最近的n条消息
        context = " ".join(history[-self.context_window:] if self.context_window > 0 else [])
        
        # 组合上下文、版本和查询生成缓存键
        combined = f"{context}|v{version}|{query}".strip()
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
        self.history_versions = {}  # 历史版本号
    
    def update_history(self, query: str, thread_id: str = "default", max_history: int = 10):
        """更新会话历史"""
        if thread_id not in self.conversation_history:
            self.conversation_history[thread_id] = []
            self.history_versions[thread_id] = 0  # 初始化版本号
        
        # 添加新查询到历史
        self.conversation_history[thread_id].append(query)
        
        # 保持历史记录在可管理的大小
        if len(self.conversation_history[thread_id]) > max_history:
            self.conversation_history[thread_id] = self.conversation_history[thread_id][-max_history:]
        
        # 增加版本号，确保上下文变化时键也会变化
        self.history_versions[thread_id] += 1
    
    def generate_key(self, query: str, **kwargs) -> str:
        """生成同时考虑上下文和关键词的缓存键"""
        thread_id = kwargs.get("thread_id", "default")
        key_parts = [query.strip()]
        
        # 添加上下文信息
        # 获取当前会话的历史记录
        history = self.conversation_history.get(thread_id, [])
        version = self.history_versions.get(thread_id, 0)
        
        # 构建上下文字符串 - 包含最近的n条消息
        if self.context_window > 0 and history:
            context = " ".join(history[-self.context_window:])
            key_parts.append(f"ctx:{hashlib.md5(context.encode('utf-8')).hexdigest()}")
        
        # 添加版本号
        key_parts.append(f"v:{version}")
        
        # 添加低级关键词（保持现有功能）
        low_level_keywords = kwargs.get("low_level_keywords", [])
        if low_level_keywords:
            # 对关键词排序，确保相同关键词集合生成相同的键
            key_parts.append("low:" + ",".join(sorted(low_level_keywords)))
        
        # 添加高级关键词（保持现有功能）
        high_level_keywords = kwargs.get("high_level_keywords", [])
        if high_level_keywords:
            # 对关键词排序，确保相同关键词集合生成相同的键
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
        self.access_times.clear()  # 确保同时清空访问时间字典
    
    def _evict_lru(self) -> None:
        """淘汰最久未使用的缓存项"""
        if not self.access_times:
            return
        
        # 找出最旧的项
        oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        self.delete(oldest_key)  # 使用delete方法确保同时清理access_times
        
    def cleanup_unused(self) -> None:
        """清理access_times中未使用的键"""
        # 找出那些在access_times中存在但在cache中不存在的键
        unused_keys = [k for k in self.access_times if k not in self.cache]
        for key in unused_keys:
            del self.access_times[key]


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
        
        successful_keys = []
        failed_keys = []
        
        for key, value in self.write_queue:
            try:
                with open(self._get_cache_path(key), 'w', encoding='utf-8') as f:
                    json.dump(value, f, ensure_ascii=False, indent=2)
                successful_keys.append(key)
            except Exception as e:
                failed_keys.append(key)
                print(f"写入缓存文件失败 ({key}): {e}")
        
        # 更新写入队列，只保留失败的项
        self.write_queue = [(k, v) for k, v in self.write_queue if k in failed_keys]
        
        # 更新时间戳
        self.last_flush_time = time.time()
        
        # 如果有成功项，保存索引
        if successful_keys:
            try:
                self._save_index()
            except Exception as e:
                print(f"保存索引失败: {e}")
                # 下次写入时会再次尝试保存索引
     
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
        self.memory_hits = 0
        self.disk_hits = 0
        self.misses = 0
        self.frequent_keys = set()  # 跟踪频繁访问的键
    
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
            # 检查是否是高质量缓存项，优先加入内存
            is_high_quality = False
            if isinstance(value, dict) and "metadata" in value:
                metadata = value.get("metadata", {})
                is_high_quality = metadata.get("user_verified", False) or metadata.get("fast_path_eligible", False)
            
            # 将磁盘中的项添加到内存缓存，优先考虑高质量项
            if is_high_quality:
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
        # 检查是否是高质量缓存项
        is_high_quality = False
        if isinstance(value, dict) and "metadata" in value:
            metadata = value.get("metadata", {})
            is_high_quality = metadata.get("user_verified", False) or metadata.get("fast_path_eligible", False)
        
        # 总是更新磁盘缓存
        self.disk_cache.set(key, value)
        
        # 高质量项总是加入内存缓存
        if is_high_quality:
            self.memory_cache.set(key, value)
        else:
            # 非高质量项根据策略决定是否加入内存
            self.memory_cache.set(key, value)
    
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
        """初始化缓存项"""
        self.content = content
        
        # 确保元数据包含必要字段
        self.metadata = metadata or {}
        
        if "created_at" not in self.metadata:
            self.metadata["created_at"] = time.time()
        if "quality_score" not in self.metadata:
            self.metadata["quality_score"] = 0
        if "user_verified" not in self.metadata:
            self.metadata["user_verified"] = False
        if "access_count" not in self.metadata:
            self.metadata["access_count"] = 0
        if "fast_path_eligible" not in self.metadata:
            # 增加快速路径标记
            self.metadata["fast_path_eligible"] = False
    
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
        """标记缓存质量"""
        if is_positive:
            current_score = self.metadata.get("quality_score", 0)
            self.metadata["quality_score"] = current_score + 1
            self.metadata["user_verified"] = True
            # 标记为适合快速路径
            self.metadata["fast_path_eligible"] = True
        else:
            current_score = self.metadata.get("quality_score", 0)
            self.metadata["quality_score"] = max(0, current_score - 2)  # 负面评价权重更大
            # 标记为不适合快速路径
            self.metadata["fast_path_eligible"] = False
    
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
        try:
            if isinstance(data, dict):
                if "content" in data and "metadata" in data:
                    # 处理完整格式
                    metadata = data["metadata"]
                    # 确保metadata具有所有必要字段
                    if not isinstance(metadata, dict):
                        metadata = {}
                    if "created_at" not in metadata:
                        metadata["created_at"] = time.time()
                    if "quality_score" not in metadata:
                        metadata["quality_score"] = 0
                    if "user_verified" not in metadata:
                        metadata["user_verified"] = False
                    if "access_count" not in metadata:
                        metadata["access_count"] = 0
                    
                    return cls(data["content"], metadata)
                else:
                    # 处理简单格式
                    return cls(data, {
                        "created_at": time.time(),
                        "quality_score": 0,
                        "user_verified": False,
                        "access_count": 0
                    })
            else:
                # 如果是非字典类型，直接作为内容
                return cls(data, {
                    "created_at": time.time(),
                    "quality_score": 0,
                    "user_verified": False,
                    "access_count": 0
                })
        except Exception as e:
            print(f"反序列化缓存项失败: {e}")
            # 返回默认对象，确保程序不会崩溃
            return cls("Error deserializing cache item", {
                "created_at": time.time(),
                "quality_score": 0,
                "user_verified": False,
                "access_count": 0,
                "error": str(e)
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
    
    
class ThreadSafeCacheBackend(CacheStorageBackend):
    """线程安全的缓存后端装饰器"""

    def __init__(self, backend: CacheStorageBackend):
        """
        初始化线程安全缓存后端
        
        参数:
            backend: 被装饰的缓存后端
        """
        self.backend = backend
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存项，线程安全"""
        with self.lock:
            return self.backend.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """设置缓存项，线程安全"""
        with self.lock:
            self.backend.set(key, value)
    
    def delete(self, key: str) -> bool:
        """删除缓存项，线程安全"""
        with self.lock:
            return self.backend.delete(key)
    
    def clear(self) -> None:
        """清空缓存，线程安全"""
        with self.lock:
            self.backend.clear()


class CacheManager:
    """统一缓存管理器，提供高级缓存功能"""
    
    def __init__(self, 
             key_strategy: CacheKeyStrategy = None, 
             storage_backend: CacheStorageBackend = None,
             cache_dir: str = "./cache",
             memory_only: bool = False,
             max_memory_size: int = 100,
             max_disk_size: int = 1000,
             thread_safe: bool = True):
        # 设置缓存键策略
        self.key_strategy = key_strategy or SimpleCacheKeyStrategy()
        
        # 设置存储后端
        backend = None
        if storage_backend:
            backend = storage_backend
        elif memory_only:
            backend = MemoryCacheBackend(max_size=max_memory_size)
        else:
            backend = HybridCacheBackend(
                cache_dir=cache_dir,
                memory_max_size=max_memory_size,
                disk_max_size=max_disk_size
            )
        
        # 如果需要线程安全，添加包装器
        if thread_safe:
            self.storage = ThreadSafeCacheBackend(backend)
        else:
            self.storage = backend
        
        # 性能指标收集
        self.performance_metrics = {}
    
    def _get_consistent_key(self, query: str, **kwargs) -> str:
        """
        生成一致的缓存键，确保不同方法间的键生成逻辑一致
        
        参数:
            query: 查询内容
            **kwargs: 其他参数，如thread_id、关键词等
            
        返回:
            str: 生成的一致缓存键
        """
        return self.key_strategy.generate_key(query, **kwargs)
    
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
        key = self._get_consistent_key(query, **kwargs)
        
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
        key = self._get_consistent_key(query, **kwargs)
        
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
        key = self._get_consistent_key(query, **kwargs)
        
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
        key = self._get_consistent_key(query, **kwargs)
        
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
        
        if is_positive and cache_item.is_high_quality():
            # 明确标记为高质量缓存，增加额外属性确保快速路径能识别
            item_dict = cache_item.to_dict()
            item_dict["metadata"]["fast_path_eligible"] = True
            self.storage.set(key, item_dict)
        
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
        key = self._get_consistent_key(query, **kwargs)
        
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