from typing import Annotated, Sequence, TypedDict, List, Dict, Any
from abc import ABC, abstractmethod
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
import pprint
import json
import hashlib
import time
import os

from model.get_models import get_llm_model

class QueryCache:
    """上下文感知查询缓存实现，支持元数据和性能优化"""
    
    def __init__(self, max_size=100, cache_dir="./cache", memory_only=False):
        self.cache = {}  # 内存缓存
        self.max_size = max_size
        self.cache_dir = cache_dir
        self.conversation_history = {}  # 存储会话历史
        self.memory_only = memory_only  # 是否仅使用内存缓存
        self.write_queue = []  # 写入队列
        self.last_flush_time = time.time()
        self.performance_metrics = {}  # 性能指标收集
        
        # 确保缓存目录存在
        if not self.memory_only and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        # 从磁盘加载缓存
        if not self.memory_only:
            self._load_from_disk()
    
    def _get_cache_key(self, query, thread_id="default", context_window=3):
        """生成上下文感知的缓存键"""
        start_time = time.time()
        
        # 获取当前会话的历史记录
        history = self.conversation_history.get(thread_id, [])
        
        # 构建上下文字符串 - 包含最近的n条消息
        context = " ".join(history[-context_window:] if context_window > 0 else [])
        
        # 组合上下文和查询生成缓存键
        combined = (context + " " + query).strip()
        key = hashlib.md5(combined.encode('utf-8')).hexdigest()
        
        # 记录性能指标
        self.performance_metrics["key_generation_time"] = time.time() - start_time
        
        return key
    
    def update_conversation_history(self, query, thread_id="default", max_history=10):
        """更新会话历史"""
        if thread_id not in self.conversation_history:
            self.conversation_history[thread_id] = []
            
        # 添加新查询到历史
        self.conversation_history[thread_id].append(query)
        
        # 保持历史记录在可管理的大小
        if len(self.conversation_history[thread_id]) > max_history:
            self.conversation_history[thread_id] = self.conversation_history[thread_id][-max_history:]
    
    def get(self, query, thread_id="default", skip_validation=False):
        """获取缓存的结果，考虑会话上下文，支持跳过验证"""
        start_time = time.time()
        
        # 生成缓存键
        key = self._get_cache_key(query, thread_id)
        
        # 检查内存缓存
        result = self.cache.get(key)
        cache_check_time = time.time() - start_time
        
        if result:
            print(f"上下文缓存命中: {query[:30]}... ({cache_check_time:.4f}s)")
            
            # 检查是否为高质量缓存
            if isinstance(result, dict) and "metadata" in result:
                metadata = result.get("metadata", {})
                
                # 已验证的高质量缓存，直接返回内容
                if metadata.get("user_verified", False) or metadata.get("quality_score", 0) > 2:
                    print(f"高质量缓存，跳过验证 (score={metadata.get('quality_score', 0)})")
                    content = result.get("content", result)
                    self.performance_metrics["total_get_time"] = time.time() - start_time
                    return content
            
            # 常规缓存，根据需要跳过验证
            if skip_validation:
                content = result.get("content", result) if isinstance(result, dict) else result
                self.performance_metrics["total_get_time"] = time.time() - start_time
                return content
                
            # 标准流程保持不变
            self.performance_metrics["total_get_time"] = time.time() - start_time
            return result
            
        self.performance_metrics["total_get_time"] = time.time() - start_time
        return None
    
    def set(self, query, result, thread_id="default", flush_threshold=10, flush_interval=30):
        """设置缓存结果，更新会话历史，支持延迟写入"""
        start_time = time.time()
        
        # 更新会话历史
        self.update_conversation_history(query, thread_id)
        
        # 获取考虑上下文的缓存键
        key = self._get_cache_key(query, thread_id)
        
        # 如果缓存已满，删除最旧的项
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            
            # 删除磁盘上的缓存文件
            if not self.memory_only:
                cache_path = self._get_cache_path(oldest_key)
                if os.path.exists(cache_path):
                    os.remove(cache_path)
        
        # 确保结果有正确的格式和元数据
        if not isinstance(result, dict) or "content" not in result:
            result = {
                "content": result,
                "metadata": {
                    "created_at": time.time(),
                    "quality_score": 0,
                    "user_verified": False,
                    "access_count": 0
                }
            }
        elif "metadata" not in result:
            result["metadata"] = {
                "created_at": time.time(),
                "quality_score": 0,
                "user_verified": False,
                "access_count": 0
            }
        
        # 更新内存缓存 - 立即执行
        self.cache[key] = result
        
        # 队列磁盘写入 - 稍后批量执行
        if not self.memory_only:
            self.write_queue.append((key, result))
            
            # 根据阈值和时间间隔决定是否刷新
            current_time = time.time()
            if (len(self.write_queue) >= flush_threshold or 
                (current_time - self.last_flush_time) > flush_interval):
                self._flush_write_queue()
        
        self.performance_metrics["set_time"] = time.time() - start_time
    
    def _flush_write_queue(self):
        """批量执行磁盘写入"""
        start_time = time.time()
        
        if not self.write_queue:
            return
            
        print(f"执行批量磁盘写入: {len(self.write_queue)} 项")
        
        for key, value in self.write_queue:
            try:
                with open(self._get_cache_path(key), 'w', encoding='utf-8') as f:
                    json.dump(value, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"写入缓存文件时出错: {e}")
        
        # 清空队列并更新时间戳
        self.write_queue = []
        self.last_flush_time = time.time()
        
        self.performance_metrics["flush_time"] = time.time() - start_time
        print(f"批量写入完成 ({self.performance_metrics['flush_time']:.4f}s)")
    
    def _get_cache_path(self, key):
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f"{key}.json")
    
    def _load_from_disk(self):
        """从磁盘加载缓存，支持带元数据的格式"""
        start_time = time.time()
        
        loaded_count = 0
        if os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                if filename.endswith(".json"):
                    key = filename[:-5]  # 移除 .json 后缀
                    try:
                        with open(os.path.join(self.cache_dir, filename), 'r', encoding='utf-8') as f:
                            content = json.load(f)
                            
                            # 兼容旧格式 - 如果是简单字符串，转换为带元数据的格式
                            if isinstance(content, str):
                                content = {
                                    "content": content,
                                    "metadata": {
                                        "created_at": time.time(),
                                        "quality_score": 0,
                                        "user_verified": False,
                                        "access_count": 0
                                    }
                                }
                                
                            self.cache[key] = content
                            loaded_count += 1
                    except Exception as e:
                        print(f"加载缓存文件 {filename} 时出错: {e}")
        
        load_time = time.time() - start_time
        print(f"从磁盘加载了 {loaded_count} 个缓存项 ({load_time:.4f}s)")
        self.performance_metrics["load_time"] = load_time
    
    def get_fast(self, query, thread_id="default"):
        """超快速缓存检索路径，仅用于高质量已验证缓存"""
        start_time = time.time()
        
        # 生成缓存键
        key = self._get_cache_key(query, thread_id)
        
        # 直接从内存缓存中检索
        result = self.cache.get(key)
        
        if result:
            # 快速检查是否为高质量已验证缓存
            if isinstance(result, dict) and "metadata" in result:
                metadata = result.get("metadata", {})
                if metadata.get("user_verified", False) or metadata.get("quality_score", 0) > 2:
                    # 高质量缓存，直接返回内容
                    fast_time = time.time() - start_time
                    print(f"快速路径命中 ({fast_time:.4f}s)")
                    
                    # 更新访问计数
                    metadata["access_count"] = metadata.get("access_count", 0) + 1
                    metadata["last_accessed"] = time.time()
                    
                    return result.get("content", "")
        
        # 未命中快速路径
        return None
    
    def mark_answer_quality(self, query, is_positive, thread_id="default"):
        """标记回答质量，用于缓存质量控制"""
        # 使用安全的缓存键
        safe_query = query.strip()
        cache_key = self._get_cache_key(safe_query, thread_id)
        
        # 从缓存中获取条目
        cached_item = self.cache.get(cache_key)
        if not cached_item:
            print(f"未找到查询的缓存: {safe_query[:30]}...")
            return
                
        # 如果是简单的字符串缓存，转换为带有元数据的格式
        if isinstance(cached_item, str):
            cached_item = {
                "content": cached_item,
                "metadata": {}
            }
        
        # 如果已经是字典，但没有元数据字段，添加它
        if "metadata" not in cached_item:
            cached_item["metadata"] = {}
                
        # 更新质量得分
        if is_positive:
            # 正面反馈增加质量得分
            current_score = cached_item["metadata"].get("quality_score", 0)
            cached_item["metadata"]["quality_score"] = current_score + 1
            cached_item["metadata"]["user_verified"] = True
        else:
            # 负面反馈降低质量得分
            current_score = cached_item["metadata"].get("quality_score", 0)
            cached_item["metadata"]["quality_score"] = max(0, current_score - 2)  # 负面反馈权重更大
                
        # 更新缓存
        self.cache[cache_key] = cached_item
        
        # 将更新添加到写入队列
        if not self.memory_only:
            self.write_queue.append((cache_key, cached_item))
            
            # 优先处理反馈更新
            self._flush_write_queue()

class BaseAgent(ABC):
    """Agent 基类，定义通用功能和接口"""
    
    def __init__(self, cache_dir="./cache", memory_only=False):
        self.llm = get_llm_model()
        self.memory = MemorySaver()
        self.execution_log = []
        self.query_cache = QueryCache(cache_dir=cache_dir, memory_only=memory_only)
        self.performance_metrics = {}  # 性能指标收集
        
        # 初始化工具
        self.tools = self._setup_tools()
        
        # 设置工作流图
        self._setup_graph()
    
    @abstractmethod
    def _setup_tools(self) -> List:
        """设置工具，子类必须实现"""
        pass
    
    def _setup_graph(self):
        """设置工作流图 - 基础结构，子类可以通过_add_retrieval_edges自定义"""
        # 定义状态类型
        class AgentState(TypedDict):
            messages: Annotated[Sequence[BaseMessage], add_messages]

        # 创建工作流图
        workflow = StateGraph(AgentState)
        
        # 添加节点 - 节点与原始代码保持一致
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("retrieve", ToolNode(self.tools))
        workflow.add_node("generate", self._generate_node)
        
        # 添加从开始到代理的边
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "retrieve",
                END: END,
            },
        )
        
        # 添加从检索到生成的边 - 这个逻辑由子类实现
        self._add_retrieval_edges(workflow)
        
        # 从生成到结束
        workflow.add_edge("generate", END)
        
        # 编译图
        self.graph = workflow.compile(checkpointer=self.memory)
    
    @abstractmethod
    def _add_retrieval_edges(self, workflow):
        """添加从检索到生成的边，子类必须实现"""
        pass
    
    def _log_execution(self, node_name: str, input_data: Any, output_data: Any):
        """记录节点执行"""
        self.execution_log.append({
            "node": node_name,
            "timestamp": time.time(),
            "input": input_data,
            "output": output_data
        })
    
    def _log_performance(self, operation, metrics):
        """记录性能指标"""
        self.performance_metrics[operation] = {
            "timestamp": time.time(),
            **metrics
        }
        
        # 输出关键性能指标
        if "duration" in metrics:
            print(f"性能指标 - {operation}: {metrics['duration']:.4f}s")
    
    def _agent_node(self, state):
        """Agent 节点逻辑"""
        messages = state["messages"]
        
        # 提取关键词优化查询
        if len(messages) > 0 and isinstance(messages[-1], HumanMessage):
            query = messages[-1].content
            keywords = self._extract_keywords(query)
            
            # 记录关键词
            self._log_execution("extract_keywords", query, keywords)
            
            # 增强消息，添加关键词信息
            if keywords:
                # 创建一个新的消息，带有关键词元数据
                enhanced_message = HumanMessage(
                    content=query,
                    additional_kwargs={"keywords": keywords}
                )
                # 替换原始消息
                messages = messages[:-1] + [enhanced_message]
        
        # 使用工具处理请求
        model = self.llm.bind_tools(self.tools)
        response = model.invoke(messages)
        
        self._log_execution("agent", messages, response)
        return {"messages": [response]}
    
    @abstractmethod
    def _extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """提取查询关键词，子类必须实现"""
        pass
    
    @abstractmethod
    def _generate_node(self, state):
        """生成回答节点逻辑，子类必须实现"""
        pass
    
    def _get_safe_cache_key(self, query):
        """生成安全的缓存键，处理中文和特殊字符"""
        return hashlib.md5(query.encode('utf-8')).hexdigest()

    def check_fast_cache(self, query: str, thread_id: str = "default") -> str:
        """专用的快速缓存检查方法，用于高性能路径"""
        start_time = time.time()
        
        # 直接使用优化缓存的快速路径
        result = self.query_cache.get_fast(query, thread_id)
        
        duration = time.time() - start_time
        self._log_performance("fast_cache_check", {
            "duration": duration,
            "hit": result is not None
        })
        
        return result
    
    def ask_with_trace(self, query: str, thread_id: str = "default", recursion_limit: int = 5) -> Dict:
        """执行查询并获取带执行轨迹的回答"""
        overall_start = time.time()
        self.execution_log = []  # 重置执行日志
        
        # 更新会话历史，用于上下文感知缓存
        self.query_cache.update_conversation_history(query, thread_id)
        
        # 检查是否有完整问答的缓存
        safe_query = query.strip()
        
        # 首先尝试快速路径 - 跳过验证的高质量缓存
        fast_cache_start = time.time()
        fast_result = self.check_fast_cache(safe_query, thread_id)
        fast_cache_time = time.time() - fast_cache_start
        
        if fast_result:
            print(f"快速路径缓存命中: {safe_query[:30]}... ({fast_cache_time:.4f}s)")
            
            return {
                "answer": fast_result,
                "execution_log": [{"node": "fast_cache_hit", "timestamp": time.time(), "input": safe_query, "output": "高质量缓存命中"}]
            }
        
        # 尝试常规缓存路径，但优化验证
        cache_start = time.time()
        cached_response = self.query_cache.get(safe_query, thread_id)
        cache_time = time.time() - cache_start
        
        if cached_response:
            print(f"完整问答缓存命中: {safe_query[:30]}... ({cache_time:.4f}s)")
            
            # 获取缓存内容
            cache_content = cached_response.get("content", cached_response) if isinstance(cached_response, dict) else cached_response
            
            return {
                "answer": cache_content,
                "execution_log": [{"node": "cache_hit", "timestamp": time.time(), "input": safe_query, "output": "常规缓存命中"}]
            }
        
        # 未命中缓存，执行标准流程
        process_start = time.time()
        
        config = {
            "configurable": {
                "thread_id": thread_id,
                "recursion_limit": recursion_limit
            }
        }
        
        inputs = {"messages": [HumanMessage(content=query)]}
        try:
            # 执行完整的处理流程
            for output in self.graph.stream(inputs, config=config):
                pprint.pprint(f"Output from node '{list(output.keys())[0]}':")
                pprint.pprint("---")
                pprint.pprint(output, indent=2, width=80, depth=None)
                pprint.pprint("\n---\n")
                
            chat_history = self.memory.get(config)["channel_values"]["messages"]
            answer = chat_history[-1].content
            
            # 准备带元数据的缓存项
            cache_item = {
                "content": answer,
                "metadata": {
                    "created_at": time.time(),
                    "quality_score": 0,
                    "user_verified": False,
                    "access_count": 1
                }
            }
            
            # 缓存处理结果 - 无需再次验证
            if answer and len(answer) > 10:
                self.query_cache.set(safe_query, cache_item, thread_id)
            
            process_time = time.time() - process_start
            print(f"完整处理耗时: {process_time:.4f}s")
            
            overall_time = time.time() - overall_start
            self._log_performance("ask_with_trace", {
                "total_duration": overall_time,
                "cache_check": cache_time,
                "processing": process_time
            })
            
            return {
                "answer": answer,
                "execution_log": self.execution_log
            }
        except Exception as e:
            error_time = time.time() - process_start
            print(f"处理查询时出错: {e} ({error_time:.4f}s)")
            return {
                "answer": f"抱歉，处理您的问题时遇到了错误。请稍后再试或换一种提问方式。错误详情: {str(e)}",
                "execution_log": self.execution_log + [{"node": "error", "timestamp": time.time(), "input": query, "output": str(e)}]
            }
    
    def ask(self, query: str, thread_id: str = "default", recursion_limit: int = 5):
        """向Agent提问"""
        overall_start = time.time()
        
        # 更新会话历史，用于上下文感知缓存
        self.query_cache.update_conversation_history(query, thread_id)
        
        # 首先尝试快速路径 - 跳过验证的高质量缓存
        fast_cache_start = time.time()
        fast_result = self.check_fast_cache(query, thread_id)
        fast_cache_time = time.time() - fast_cache_start
        
        if fast_result:
            print(f"快速路径缓存命中: {query[:30]}... ({fast_cache_time:.4f}s)")
            return fast_result
        
        # 尝试常规缓存路径，但优化验证
        cache_start = time.time()
        cached_response = self.query_cache.get(query, thread_id, skip_validation=True)
        cache_time = time.time() - cache_start
        
        if cached_response:
            print(f"常规缓存命中，跳过验证: {query[:30]}... ({cache_time:.4f}s)")
            
            # 获取实际内容
            if isinstance(cached_response, dict) and "content" in cached_response:
                return cached_response["content"]
            return cached_response
        
        # 未命中缓存，执行标准流程
        process_start = time.time()
        
        # 正常处理请求
        config = {
            "configurable": {
                "thread_id": thread_id,
                "recursion_limit": recursion_limit
            }
        }
        
        inputs = {"messages": [HumanMessage(content=query)]}
        try:
            for output in self.graph.stream(inputs, config=config):
                pass
                
            chat_history = self.memory.get(config)["channel_values"]["messages"]
            answer = chat_history[-1].content
            
            # 准备带元数据的缓存项 - 直接设置为有效
            cache_item = {
                "content": answer,
                "metadata": {
                    "created_at": time.time(),
                    "quality_score": 1,  # 默认给一个基础分
                    "user_verified": False,
                    "access_count": 1
                }
            }
            
            # 缓存处理结果 - 直接缓存，无需验证
            if answer and len(answer) > 10:
                self.query_cache.set(query, cache_item, thread_id)
            
            process_time = time.time() - process_start
            overall_time = time.time() - overall_start
            
            self._log_performance("ask", {
                "total_duration": overall_time,
                "cache_check": cache_time,
                "processing": process_time
            })
            
            return answer
        except Exception as e:
            error_time = time.time() - process_start
            print(f"处理查询时出错: {e} ({error_time:.4f}s)")
            return f"抱歉，处理您的问题时遇到了错误。请稍后再试或换一种提问方式。错误详情: {str(e)}"

    
    def mark_answer_quality(self, query: str, is_positive: bool, thread_id: str = "default"):
        """标记回答质量，用于缓存质量控制 - 优化版"""
        start_time = time.time()
        
        # 直接调用优化缓存的质量标记方法
        self.query_cache.mark_answer_quality(query, is_positive, thread_id)
        
        mark_time = time.time() - start_time
        self._log_performance("mark_quality", {
            "duration": mark_time,
            "is_positive": is_positive
        })
    
    def clear_cache_for_query(self, query: str, thread_id: str = "default"):
        """清除特定查询的缓存"""
        # 使用上下文感知的缓存键
        safe_query = query.strip()
        cache_key = self.query_cache._get_cache_key(safe_query, thread_id)
        
        # 从缓存中删除
        if cache_key in self.query_cache.cache:
            del self.query_cache.cache[cache_key]
            print(f"已从内存缓存中清除查询: {safe_query[:30]}...")
            
        # 从磁盘中删除
        if not self.query_cache.memory_only:
            cache_path = self.query_cache._get_cache_path(cache_key)
            if os.path.exists(cache_path):
                os.remove(cache_path)
                print(f"已从磁盘缓存中清除查询: {safe_query[:30]}...")
            
        return True

    def _validate_answer(self, query: str, answer: str, thread_id: str = "default") -> bool:
        """验证答案质量，整合用户反馈信息
        
        改进的验证方法，考虑用户反馈:
        1. 如果答案已被用户验证为高质量，直接返回True
        2. 如果答案已被用户标记为低质量，直接返回False
        3. 否则进行常规的质量检查
        """
        # 检查是否有缓存及其用户反馈
        cache_key = self.query_cache._get_cache_key(query.strip(), thread_id)
        cached_item = self.query_cache.cache.get(cache_key)
        
        if cached_item and isinstance(cached_item, dict) and "metadata" in cached_item:
            # 检查是否有用户验证
            if cached_item["metadata"].get("user_verified", False):
                # 用户已验证为好答案
                return True
                
            # 检查质量分数
            quality_score = cached_item["metadata"].get("quality_score", 0)
            if quality_score < 0:
                # 用户给了负面评价
                return False
        
        # 如果没有用户反馈或反馈不明确，执行常规验证
        # 基本检查 - 长度
        if len(answer) < 20:
            return False
            
        # 检查是否包含错误消息
        error_patterns = [
            "抱歉，处理您的问题时遇到了错误",
            "技术原因:",
            "无法获取",
            "无法回答这个问题"
        ]
        
        for pattern in error_patterns:
            if pattern in answer:
                return False
                
        # 相关性检查 - 检查问题关键词是否在答案中出现
        keywords = self._extract_keywords(query)
        if keywords:
            low_level_keywords = keywords.get("low_level", [])
            if low_level_keywords:
                # 至少有一个低级关键词应该在答案中出现
                keyword_found = any(keyword.lower() in answer.lower() for keyword in low_level_keywords)
                if not keyword_found:
                    return False
        
        # 通过所有检查
        return True
    
    def close(self):
        """关闭资源"""
        # 确保所有延迟写入的缓存项都被保存
        if hasattr(self.query_cache, '_flush_write_queue'):
            self.query_cache._flush_write_queue()