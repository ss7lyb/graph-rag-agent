# 使用 hybrid search tool 的agent
from typing import Annotated, Sequence, TypedDict, List, Dict, Any
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
import time

from model.get_models import get_llm_model
from config.prompt import LC_SYSTEM_PROMPT
from config.settings import response_type
from search.tool.hybrid_tool import EnhancedSearchTool

class GraphAgent:
    def __init__(self):
        self.llm = get_llm_model()
        self.memory = MemorySaver()
        self.execution_log = []
        
        # 初始化增强型搜索工具
        self.search_tool = EnhancedSearchTool()
        self.tools = [
            self.search_tool.get_tool(),
            self.search_tool.get_global_tool(),
        ]
        
        self._setup_graph()

    def _setup_graph(self):
        """Setup the workflow graph"""
        # Define state type
        class AgentState(TypedDict):
            messages: Annotated[Sequence[BaseMessage], add_messages]

        # Create workflow graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("retrieve", ToolNode(self.tools))
        workflow.add_node("generate", self._generate_node)
        
        # Add edges
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "retrieve",
                END: END,
            },
        )
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        # Compile graph
        self.graph = workflow.compile(checkpointer=self.memory)

    def _log_execution(self, node_name: str, input_data: Any, output_data: Any):
        """Log the execution of a node"""
        self.execution_log.append({
            "node": node_name,
            "timestamp": time.time(),
            "input": input_data,
            "output": output_data
        })

    def _agent_node(self, state):
        """Agent node logic"""
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

    def _extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """提取查询关键词"""
        # 使用增强型搜索工具的关键词提取功能
        return self.search_tool.extract_keywords(query)

    def _generate_node(self, state):
        """Generate answer node logic"""
        messages = state["messages"]
        question = messages[-3].content
        docs = messages[-1].content

        # 使用增强型提示模板
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
        response = rag_chain.invoke({
            "context": docs, 
            "question": question, 
            "response_type": response_type
        })
        
        self._log_execution("generate", 
                           {"question": question, "docs_length": len(docs)}, 
                           response)
        
        return {"messages": [AIMessage(content=response)]}

    def ask_with_trace(self, query: str, thread_id: str = "default", recursion_limit: int = 5) -> Dict:
        """Ask a question and get both the answer and execution trace"""
        self.execution_log = []  # Reset execution log
        
        config = {
            "configurable": {
                "thread_id": thread_id,
                "recursion_limit": recursion_limit
            }
        }
        
        inputs = {"messages": [HumanMessage(content=query)]}
        for output in self.graph.stream(inputs, config=config):
            pass
            
        chat_history = self.memory.get(config)["channel_values"]["messages"]
        answer = chat_history[-1].content
        
        return {
            "answer": answer,
            "execution_log": self.execution_log
        }

    def ask(self, query: str, thread_id: str = "default", recursion_limit: int = 5):
        """Ask a question to the agent"""
        config = {
            "configurable": {
                "thread_id": thread_id,
                "recursion_limit": recursion_limit
            }
        }
        
        inputs = {"messages": [HumanMessage(content=query)]}
        for output in self.graph.stream(inputs, config=config):
            pass
            
        chat_history = self.memory.get(config)["channel_values"]["messages"]
        return chat_history[-1].content
        
    def close(self):
        """关闭资源"""
        if self.search_tool:
            self.search_tool.close()

if __name__ == "__main__":
    agent = GraphAgent()
    
    try:
        # 测试基本问答
        print(agent.ask("你好，想问一些问题。"))
        print(agent.ask("描述一下悟空第一次见到菩提祖师的场景？"))
        print(agent.ask("《悟空传》的主要人物有哪些？"))
        print(agent.ask("他们最后的结局是什么？"))
    finally:
        # 确保资源被正确关闭
        agent.close()