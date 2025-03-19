from typing import List, Dict, Generator
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.prompt import LC_SYSTEM_PROMPT
from config.settings import response_type
from search.tool.deep_research_tool import DeepResearchTool

from agent.base import BaseAgent


class DeepResearchAgent(BaseAgent):
    """
    深度研究代理：使用DeepResearcher实现多步骤推理的代理
    
    该代理扩展了基础代理架构，使用多回合的思考、搜索和推理来解决复杂问题。
    主要特点：
    1. 显式推理过程
    2. 迭代式搜索
    3. 高质量知识整合
    4. 支持流式输出
    """
    
    def __init__(self):
        # 初始化深度研究工具 - 支持流式输出
        self.research_tool = DeepResearchTool()
        
        # 设置缓存目录
        self.cache_dir = "./cache/deep_research_agent"
        
        # 设置查看推理过程的模式
        self.show_thinking = False

        # 调用父类构造函数
        super().__init__(cache_dir=self.cache_dir)
    
    def _setup_chains(self):
        # DeepResearchTool 主要通过 thinking 方法和其他工具来处理查询
        # 不需要处理链
        pass
    
    def _setup_tools(self) -> List:
        """设置工具"""
        tools = []
        
        # 根据模式选择不同的工具
        if self.show_thinking:
            # 思考过程可见模式
            tools.append(self.research_tool.get_thinking_tool())
        else:
            # 标准模式
            tools.append(self.research_tool.get_tool())
            
        return tools
    
    def _add_retrieval_edges(self, workflow):
        """添加从检索到生成的边"""
        # 简单的从检索直接到生成
        workflow.add_edge("retrieve", "generate")
    
    def _extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """从查询中提取关键词"""
        # 使用研究工具的关键词提取功能
        return self.research_tool.extract_keywords(query)
    
    def _generate_node(self, state):
        """生成回答节点逻辑"""
        messages = state["messages"]
        
        # 安全地获取问题和检索结果
        try:
            # 原始问题在倒数第三个消息
            question = messages[-3].content if len(messages) >= 3 else "未找到问题"
            # 检索结果在最后一个消息
            retrieval_result = messages[-1].content if messages[-1] else "未找到相关信息"
        except Exception as e:
            return {"messages": [AIMessage(content=f"生成回答时出错: {str(e)}")]}

        # 获取当前会话ID
        thread_id = state.get("configurable", {}).get("thread_id", "default")
            
        # 检查缓存
        cached_result = self.cache_manager.get(f"generate:{question}", thread_id=thread_id)
        if cached_result:
            return {"messages": [AIMessage(content=cached_result)]}

        # 检查流式输出的情况 - 流式结果通常是生成器或特殊格式
        if isinstance(retrieval_result, (Generator, dict)) and not isinstance(retrieval_result, str):
            # 如果是流式结果，直接返回，外部处理
            return {"messages": [AIMessage(content=retrieval_result)]}

        # 如果检索结果不是思考过程（当使用非思考工具时）
        if not isinstance(retrieval_result, str) or not retrieval_result.startswith("<think>"):
            # 直接返回检索结果
            if self.cache_manager.validate_answer(question, retrieval_result):
                self.cache_manager.set(f"generate:{question}", retrieval_result, thread_id=thread_id)
            return {"messages": [AIMessage(content=retrieval_result)]}
        
        # 处理思考过程（当使用思考工具时）
        try:
            # 提取思考过程
            thinking = retrieval_result
            
            # 创建总结提示
            prompt = ChatPromptTemplate.from_messages([
                ("system", LC_SYSTEM_PROMPT),
                ("human", """
                    以下是对问题的思考过程:
                    
                    {thinking}
                    
                    原始问题是:
                    {question}
                    
                    请生成一个全面、有深度的回答，不要重复思考过程，直接给出最终综合结论。
                    """),
            ])
            
            # 创建处理链
            chain = prompt | self.llm | StrOutputParser()
            
            # 生成回答
            response = chain.invoke({
                "thinking": thinking,
                "question": question,
                "response_type": response_type
            })
            
            # 缓存结果
            if self.cache_manager.validate_answer(question, response):
                self.cache_manager.set(f"generate:{question}", response, thread_id=thread_id)
            
            return {"messages": [AIMessage(content=response)]}
            
        except Exception as e:
            error_msg = f"处理思考过程时出错: {str(e)}"
            return {"messages": [AIMessage(content=error_msg)]}
    
    def ask(self, query: str, thread_id: str = "default", recursion_limit: int = 5, show_thinking: bool = False):
        """
        向Agent提问，可选显示思考过程
        
        参数:
            query: 用户问题
            thread_id: 会话ID
            recursion_limit: 递归限制
            show_thinking: 是否显示思考过程
                
        返回:
            str: 生成的回答或包含思考过程的字典
        """
        # 设置是否显示思考过程
        old_thinking = self.show_thinking
        self.show_thinking = show_thinking
        
        try:
            # 调用父类方法
            result = super().ask(query, thread_id, recursion_limit)
            return result
        finally:
            # 重置状态
            self.show_thinking = old_thinking
    
    def ask_with_thinking(self, query: str, thread_id: str = "default"):
        """
        提问并返回带思考过程的答案
        
        参数:
            query: 用户问题
            thread_id: 会话ID
            
        返回:
            dict: 包含思考过程和答案的字典
        """
        # 直接调用研究工具的thinking方法
        result = self.research_tool.thinking(query)
        
        # 确保结果包含执行日志
        if "execution_logs" not in result:
            result["execution_logs"] = []
            
        return result