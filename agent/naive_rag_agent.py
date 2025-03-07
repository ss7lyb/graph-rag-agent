from typing import List, Dict
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.prompt import NAIVE_PROMPT
from config.settings import response_type
from search.tool.naive_search_tool import NaiveSearchTool
from agent.base import BaseAgent


class NaiveRagAgent(BaseAgent):
    """使用简单向量检索的Naive RAG Agent实现"""
    
    def __init__(self):
        # 初始化Naive搜索工具
        self.search_tool = NaiveSearchTool()
        
        # 设置缓存目录
        self.cache_dir = "./cache/naive_agent"
        
        # 调用父类构造函数
        super().__init__(cache_dir=self.cache_dir)

    def _setup_tools(self) -> List:
        """设置工具"""
        return [
            self.search_tool.get_tool(),
        ]
    
    def _add_retrieval_edges(self, workflow):
        """添加从检索到生成的边"""
        # 简单的从检索直接到生成，无需复杂路由
        workflow.add_edge("retrieve", "generate")

    def _extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        提取查询关键词 - 简化版本，不做实际的关键词提取
        
        参数:
            query: 查询字符串
            
        返回:
            Dict[str, List[str]]: 关键词字典，包含低级和高级关键词（空列表）
        """
        # Naive实现不需要关键词提取
        return {"low_level": [], "high_level": []}

    def _generate_node(self, state):
        """生成回答节点逻辑"""
        messages = state["messages"]
        
        # 安全地获取问题和检索结果
        try:
            question = messages[-3].content if len(messages) >= 3 else "未找到问题"
        except Exception:
            question = "无法获取问题"
            
        try:
            docs = messages[-1].content if messages[-1] else "未找到相关信息"
        except Exception:
            docs = "无法获取检索结果"

        # 获取当前会话ID
        thread_id = state.get("configurable", {}).get("thread_id", "default")
            
        # 检查缓存
        cached_result = self.cache_manager.get(f"generate:{question}", thread_id=thread_id)
        if cached_result:
            self._log_execution("generate", 
                               {"question": question, "docs_length": len(docs)}, 
                               "缓存命中")
            return {"messages": [AIMessage(content=cached_result)]}

        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages([
        ("system", NAIVE_PROMPT),
        ("human", """
            ---检索结果--- 
            {context}
            
            问题：
            {question}
            """),
        ])

        rag_chain = prompt | self.llm | StrOutputParser()
        try:
            response = rag_chain.invoke({
                "context": docs, 
                "question": question, 
                "response_type": response_type
            })
            
            # 缓存结果
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