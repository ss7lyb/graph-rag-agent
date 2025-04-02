from typing import List, Dict
import time
import re
import asyncio

from agent.base import BaseAgent
from agent.agent_coordinator import GraphRAGAgentCoordinator
from search.tool.local_search_tool import LocalSearchTool
from search.tool.global_search_tool import GlobalSearchTool
from search.tool.deeper_research_tool import DeeperResearchTool
from search.tool.reasoning.chain_of_exploration import ChainOfExplorationSearcher
from model.get_models import get_embeddings_model

class FusionGraphRAGAgent(BaseAgent):
    """
    Fusion GraphRAG Agent
    
    基于多Agent协作架构的增强型GraphRAGAgent，集成了多种搜索策略和知识融合方法。
    提供图谱感知、社区结构、Chain of Exploration等高级功能，实现更深度的知识检索和推理。
    """
    
    def __init__(self):
        """初始化Fusion GraphRAG Agent"""
        # 设置缓存目录
        self.cache_dir = "./cache/fusion_graphrag"
        
        # 调用父类构造函数
        super().__init__(cache_dir=self.cache_dir)
        
        # 创建协调器
        self.coordinator = GraphRAGAgentCoordinator(self.llm)
        
        # 初始化基础搜索工具 - 用于关键词提取
        self.search_tool = LocalSearchTool()
    
    def _setup_tools(self) -> List:
        """设置工具"""
        # 创建工具实例
        self.local_tool = LocalSearchTool()
        self.global_tool = GlobalSearchTool()
        self.research_tool = DeeperResearchTool()
        
        # 创建Chain of Exploration搜索器
        from config.neo4jdb import get_db_manager
        db_manager = get_db_manager()
        graph = db_manager.get_graph()
        self.chain_explorer = ChainOfExplorationSearcher(
            graph, self.llm, get_embeddings_model()
        )
        
        # 返回协调器使用的工具
        return [
            self.local_tool.get_tool(),
            self.global_tool.get_tool(),
            self.research_tool.get_tool()
        ]
    
    def _add_retrieval_edges(self, workflow):
        """添加从检索到生成的边"""
        # 简单的从检索直接到生成，具体逻辑由协调器处理
        workflow.add_edge("retrieve", "generate")
    
    def _extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """提取查询关键词"""
        # 使用搜索工具提取关键词
        return self.search_tool.extract_keywords(query)
    
    def _generate_node(self, state):
        """生成回答节点逻辑"""
        # 获取消息
        messages = state["messages"]
        
        # 安全获取问题内容
        try:
            question = messages[-3].content if len(messages) >= 3 else "未找到问题"
        except Exception:
            question = "无法获取问题"
        
        # 获取检索结果（或工具调用结果）
        try:
            tool_results = messages[-1].content if messages[-1] else "未找到相关信息"
        except Exception:
            tool_results = "无法获取检索结果"

        # 首先尝试全局缓存
        global_result = self.global_cache_manager.get(question)
        if global_result:
            self._log_execution("generate", 
                            {"question": question, "results_length": len(str(tool_results))}, 
                            "全局缓存命中")
            return {"messages": [{"role": "assistant", "content": global_result}]}
            
        # 获取当前会话ID
        thread_id = state.get("configurable", {}).get("thread_id", "default")
            
        # 然后检查会话缓存
        cached_result = self.cache_manager.get(question, thread_id=thread_id)
        if cached_result:
            self._log_execution("generate", 
                            {"question": question, "results_length": len(str(tool_results))}, 
                            "会话缓存命中")
            # 将命中内容同步到全局缓存
            self.global_cache_manager.set(question, cached_result)
            return {"messages": [{"role": "assistant", "content": cached_result}]}
        
        # 使用协调器处理
        try:
            start_time = time.time()
            
            # 使用协调器处理查询，对工具结果进行整合
            result = self.coordinator.process_query(question)
            answer = result.get("answer", "未能生成回答")
            
            # 记录性能
            self._log_performance("coordinator_process", {
                "duration": time.time() - start_time,
                "metrics": result.get("metrics", {})
            })
            
            # 缓存结果 - 同时更新会话缓存和全局缓存
            # 更新会话缓存
            self.cache_manager.set(question, answer, thread_id=thread_id)
            # 更新全局缓存
            self.global_cache_manager.set(question, answer)
            
            self._log_execution("generate", 
                            {"question": question, "results_length": len(str(tool_results))}, 
                            answer)
            
            return {"messages": [{"role": "assistant", "content": answer}]}
            
        except Exception as e:
            error_msg = f"生成回答时出错: {str(e)}"
            self._log_execution("generate_error", 
                            {"question": question, "results_length": len(str(tool_results))}, 
                            error_msg)
            return {"messages": [{"role": "assistant", "content": f"抱歉，我无法回答这个问题。技术原因: {str(e)}"}]}
    
    async def _stream_process(self, inputs, config):
        """流式处理过程"""
        # 获取查询
        query = inputs["messages"][-1].content if len(inputs["messages"]) > 0 else ""
        if not query:
            yield "无法获取查询内容，请重试。"
            return
            
        # 获取会话ID
        thread_id = config.get("configurable", {}).get("thread_id", "default")
            
        # 检查缓存
        cached_result = self.cache_manager.get(query.strip(), thread_id=thread_id)
        if cached_result:
            self._log_execution("stream_cache_hit", {"query": query}, "缓存命中")
            # 分块返回缓存结果
            sentences = re.split(r'([.!?。！？]\s*)', cached_result)
            buffer = ""
            
            for i in range(0, len(sentences)):
                buffer += sentences[i]
                
                # 当缓冲区包含完整句子或达到合理大小时输出
                if (i % 2 == 1) or len(buffer) >= 40:
                    yield buffer
                    buffer = ""
                    await asyncio.sleep(0.01)
            
            # 输出任何剩余内容
            if buffer:
                yield buffer
            return
            
        # 使用协调器的流式处理
        try:
            self._log_execution("stream_process_start", {"query": query}, "开始流式处理")
            async for chunk in self.coordinator.process_query_stream(query):
                yield chunk
            
            # 注意：协调器内部会处理缓存
            self._log_execution("stream_process_complete", {"query": query}, "完成流式处理")
            
        except Exception as e:
            error_msg = f"处理查询时出错: {str(e)}"
            self._log_execution("stream_process_error", {"query": query}, error_msg)
            yield f"**处理查询时出错**: {str(e)}"
    
    def close(self):
        """关闭资源"""
        # 调用父类方法
        super().close()
        
        # 关闭各种工具资源
        if hasattr(self, 'local_tool'):
            self.local_tool.close()
        if hasattr(self, 'global_tool'):
            self.global_tool.close()
        if hasattr(self, 'research_tool'):
            self.research_tool.close()