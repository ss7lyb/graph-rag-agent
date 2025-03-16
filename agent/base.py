from typing import Annotated, Sequence, TypedDict, List, Dict, Any
from abc import ABC, abstractmethod
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
import pprint
import time

from model.get_models import get_llm_model
from CacheManage.manager import (
    CacheManager, 
    ContextAwareCacheKeyStrategy, 
    HybridCacheBackend
)

class BaseAgent(ABC):
    """Agent 基类，定义通用功能和接口"""
    
    def __init__(self, cache_dir="./cache", memory_only=False):
        self.llm = get_llm_model()
        self.memory = MemorySaver()
        self.execution_log = []
        
        # 使用新的缓存框架
        self.cache_manager = CacheManager(
            key_strategy=ContextAwareCacheKeyStrategy(),
            storage_backend=HybridCacheBackend(
                cache_dir=cache_dir,
                memory_max_size=200,
                disk_max_size=2000
            ) if not memory_only else None,
            cache_dir=cache_dir,
            memory_only=memory_only
        )
        
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
    
    def check_fast_cache(self, query: str, thread_id: str = "default") -> str:
        """专用的快速缓存检查方法，用于高性能路径"""
        start_time = time.time()
        
        # 提取关键词，确保在缓存键中使用
        keywords = self._extract_keywords(query)
        cache_params = {
            "thread_id": thread_id,
            "low_level_keywords": keywords.get("low_level", []),
            "high_level_keywords": keywords.get("high_level", [])
        }
        
        # 使用缓存管理器的快速获取方法，传递相关参数
        result = self.cache_manager.get_fast(query, **cache_params)
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
        
        # 确保查询字符串是干净的
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
        
        # 尝试常规缓存路径
        cache_start = time.time()
        cached_response = self.cache_manager.get(safe_query, thread_id=thread_id)
        cache_time = time.time() - cache_start
        
        if cached_response:
            print(f"完整问答缓存命中: {safe_query[:30]}... ({cache_time:.4f}s)")
            
            return {
                "answer": cached_response,
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
            
            # 缓存处理结果
            if answer and len(answer) > 10:
                self.cache_manager.set(safe_query, answer, thread_id=thread_id)
            
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
        
        # 确保查询字符串是干净的
        safe_query = query.strip()
        
        # 首先尝试快速路径 - 跳过验证的高质量缓存
        fast_cache_start = time.time()
        fast_result = self.check_fast_cache(safe_query, thread_id)
        fast_cache_time = time.time() - fast_cache_start
        
        if fast_result:
            print(f"快速路径缓存命中: {safe_query[:30]}... ({fast_cache_time:.4f}s)")
            return fast_result
        
        # 尝试常规缓存路径，但优化验证
        cache_start = time.time()
        cached_response = self.cache_manager.get(safe_query, skip_validation=True, thread_id=thread_id)
        cache_time = time.time() - cache_start
        
        if cached_response:
            print(f"常规缓存命中，跳过验证: {safe_query[:30]}... ({cache_time:.4f}s)")
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
            
            # 缓存处理结果
            if answer and len(answer) > 10:
                self.cache_manager.set(safe_query, answer, thread_id=thread_id)
            
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
        """标记回答质量，用于缓存质量控制"""
        start_time = time.time()
        
        # 提取关键词
        keywords = self._extract_keywords(query)
        cache_params = {
            "thread_id": thread_id,
            "low_level_keywords": keywords.get("low_level", []),
            "high_level_keywords": keywords.get("high_level", [])
        }
        
        # 调用缓存管理器的质量标记方法，传递相关参数
        marked = self.cache_manager.mark_quality(query.strip(), is_positive, **cache_params)
        
        mark_time = time.time() - start_time
        self._log_performance("mark_quality", {
            "duration": mark_time,
            "is_positive": is_positive
        })
    
    def clear_cache_for_query(self, query: str, thread_id: str = "default"):
        """清除特定查询的缓存"""
        # 提取关键词
        keywords = self._extract_keywords(query)
        cache_params = {
            "thread_id": thread_id,
            "low_level_keywords": keywords.get("low_level", []),
            "high_level_keywords": keywords.get("high_level", [])
        }
        
        # 调用缓存管理器的删除方法，传递相关参数
        return self.cache_manager.delete(query.strip(), **cache_params)
    
    def _validate_answer(self, query: str, answer: str, thread_id: str = "default") -> bool:
        """验证答案质量"""
        
        # 使用缓存管理器的验证方法
        def validator(query, answer):
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
        
        return self.cache_manager.validate_answer(query, answer, validator, thread_id=thread_id)
    
    def close(self):
        """关闭资源"""
        # 确保所有延迟写入的缓存项都被保存
        if hasattr(self.cache_manager.storage, '_flush_write_queue'):
            self.cache_manager.storage._flush_write_queue()