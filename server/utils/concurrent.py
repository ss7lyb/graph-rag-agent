import threading
import time
from typing import Dict


class ConcurrentManager:
    """并发管理器，用于处理请求锁和超时清理"""
    
    def __init__(self, timeout_seconds=300):
        """
        初始化并发管理器
        
        Args:
            timeout_seconds: 锁超时时间(秒)
        """
        self.locks: Dict[str, threading.Lock] = {}
        self.timestamps: Dict[str, float] = {}
        self.timeout_seconds = timeout_seconds
    
    def get_lock(self, key: str) -> threading.Lock:
        """
        获取指定键的锁
        
        Args:
            key: 锁键名
            
        Returns:
            threading.Lock: 线程锁对象
        """
        if key not in self.locks:
            self.locks[key] = threading.Lock()
            self.timestamps[key] = time.time()
        return self.locks[key]
    
    def update_timestamp(self, key: str) -> None:
        """
        更新指定键的时间戳
        
        Args:
            key: 锁键名
        """
        self.timestamps[key] = time.time()
    
    def try_acquire_lock(self, key: str) -> bool:
        """
        非阻塞方式尝试获取锁
        
        Args:
            key: 锁键名
            
        Returns:
            bool: 是否成功获取锁
        """
        lock = self.get_lock(key)
        return lock.acquire(blocking=False)
    
    def release_lock(self, key: str) -> None:
        """
        释放指定键的锁
        
        Args:
            key: 锁键名
        """
        if key in self.locks and self.locks[key].locked():
            self.locks[key].release()
    
    def cleanup_expired_locks(self) -> None:
        """清理过期的锁"""
        current_time = time.time()
        expired_keys = []
        
        for key, timestamp in self.timestamps.items():
            if current_time - timestamp > self.timeout_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            if key in self.locks:
                try:
                    if not self.locks[key].locked():
                        del self.locks[key]
                except:
                    pass  # 忽略删除锁时的错误
            if key in self.timestamps:
                del self.timestamps[key]


# 创建全局实例
chat_manager = ConcurrentManager()
feedback_manager = ConcurrentManager()