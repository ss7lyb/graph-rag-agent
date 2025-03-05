from typing import List, Dict
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.prompt import LC_SYSTEM_PROMPT
from config.settings import response_type
from search.tool.hybrid_tool import HybridSearchTool

from agent.base import BaseAgent


class HybridAgent(BaseAgent):
    """使用混合搜索的Agent实现"""
    
    def __init__(self):
        # 初始化混合搜索工具
        self.search_tool = HybridSearchTool()
        
        # 首先初始化基础属性
        self.cache_dir = "./cache/hybrid_agent"
        
        # 调用父类构造函数 - 使用默认的ContextAwareCacheKeyStrategy
        super().__init__(cache_dir=self.cache_dir)

    def _setup_tools(self) -> List:
        """设置工具"""
        return [
            self.search_tool.get_tool(),
            self.search_tool.get_global_tool(),
        ]
    
    def _add_retrieval_edges(self, workflow):
        """添加从检索到生成的边"""
        # 简单的从检索直接到生成
        workflow.add_edge("retrieve", "generate")

    def _extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """提取查询关键词"""
        # 检查缓存
        cached_keywords = self.cache_manager.get(f"keywords:{query}")
        if cached_keywords:
            return cached_keywords
            
        try:
            # 使用增强型搜索工具的关键词提取功能
            keywords = self.search_tool.extract_keywords(query)
            
            # 确保返回有效的关键词格式
            if not isinstance(keywords, dict):
                keywords = {}
            if "low_level" not in keywords:
                keywords["low_level"] = []
            if "high_level" not in keywords:
                keywords["high_level"] = []
            
            # 缓存结果
            self.cache_manager.set(f"keywords:{query}", keywords)
            
            return keywords
        except Exception as e:
            print(f"关键词提取失败: {e}")
            # 出错时返回默认空关键词
            return {"low_level": [], "high_level": []}

    def _generate_node(self, state):
        """生成回答节点逻辑"""
        messages = state["messages"]
        
        # 安全地获取问题内容
        try:
            question = messages[-3].content if len(messages) >= 3 else "未找到问题"
        except Exception:
            question = "无法获取问题"
            
        # 安全地获取文档内容
        try:
            docs = messages[-1].content if messages[-1] else "未找到相关信息"
        except Exception:
            docs = "无法获取检索结果"

        # 获取当前会话ID，用于上下文感知缓存
        thread_id = state.get("configurable", {}).get("thread_id", "default")
            
        # 检查缓存 - 仅传递thread_id，让默认的上下文感知策略生效
        cached_result = self.cache_manager.get(f"generate:{question}", thread_id=thread_id)
        if cached_result:
            self._log_execution("generate", 
                               {"question": question, "docs_length": len(docs)}, 
                               "缓存命中")
            return {"messages": [AIMessage(content=cached_result)]}

        prompt = ChatPromptTemplate.from_messages([
        ("system", LC_SYSTEM_PROMPT),
        ("human", """
            ---分析报告--- 
            以下是检索到的相关信息，按重要性排序：
            
            {context}
            
            用户的问题是：
            {question}
            
            请以清晰、全面的方式回答问题，确保：
            1. 回答结合了检索到的低级（实体细节）和高级（主题概念）信息
            2. 使用三级标题(###)组织内容，增强可读性
            3. 结尾处用"#### 引用数据"标记引用来源
            """),
        ])

        rag_chain = prompt | self.llm | StrOutputParser()
        try:
            response = rag_chain.invoke({
                "context": docs, 
                "question": question, 
                "response_type": response_type
            })
            
            # 只缓存有效的生成结果 - 仅传递thread_id
            if response and len(response) > 10:
                self.cache_manager.set(f"generate:{question}", response, thread_id=thread_id)
            
            self._log_execution("generate", 
                              {"question": question, "docs_length": len(docs)}, 
                              response)
            
            return {"messages": [AIMessage(content=response)]}
        except Exception as e:
            error_msg = f"生成回答时出错: {str(e)}"
            self._log_execution("generate_error", 
                              {"question": question, "docs_length": len(docs)}, 
                              error_msg)
            return {"messages": [AIMessage(content=f"抱歉，我无法回答这个问题。技术原因: {str(e)}")]}
    
    def close(self):
        """关闭资源"""
        # 先关闭父类资源
        super().close()
        
        # 再关闭搜索工具资源
        if self.search_tool:
            self.search_tool.close()