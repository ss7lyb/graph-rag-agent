from typing import Annotated, Literal, Sequence, TypedDict, List, Dict, Any
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
import pprint

from model.get_models import get_llm_model
from config.prompt import LC_SYSTEM_PROMPT, REDUCE_SYSTEM_PROMPT
from config.settings import response_type
from search.tool.local_search_tool import LocalSearchTool
from search.tool.global_search_tool import GlobalSearchTool

class GraphAgent:
    def __init__(self):
        self.llm = get_llm_model()
        self.memory = MemorySaver()
        self.execution_log = []
        
        # Setup tools
        self.local_tool = LocalSearchTool()
        self.global_tool = GlobalSearchTool()
        self.tools = [
            self.local_tool.get_tool(),
            self.global_tool.search,
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
        workflow.add_node("rewrite", self._rewrite_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("reduce", self._reduce_node)
        
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
        workflow.add_conditional_edges(
            "retrieve",
            self._grade_documents,
        )
        workflow.add_edge("generate", END)
        workflow.add_edge("rewrite", "agent")
        workflow.add_edge("reduce", END)
        
        # Compile graph
        self.graph = workflow.compile(checkpointer=self.memory)

    def _log_execution(self, node_name: str, input_data: Any, output_data: Any):
        """Log the execution of a node"""
        self.execution_log.append({
            "node": node_name,
            "input": input_data,
            "output": output_data
        })

    def _agent_node(self, state):
        """Agent node logic"""
        messages = state["messages"]
        model = self.llm.bind_tools(self.tools)
        response = model.invoke(messages)
        
        self._log_execution("agent", messages, response)
        return {"messages": [response]}

    def _grade_documents(self, state) -> Literal["generate", "rewrite", "reduce"]:
        """Grade documents relevance"""
        messages = state["messages"]
        retrieve_message = messages[-2]
        
        # 安全地检查是否为全局检索工具调用
        tool_calls = retrieve_message.additional_kwargs.get("tool_calls", [])
        if tool_calls and tool_calls[0].get("function", {}).get("name") == "global_retriever":
            self._log_execution("grade_documents", messages, "reduce")
            return "reduce"

        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
            input_variables=["context", "question"],
        )
        
        chain = prompt | self.llm
        question = messages[-3].content
        docs = messages[-1].content
        
        score = chain.invoke({"question": question, "context": docs}).content
        result = "generate" if score.lower() == "yes" else "rewrite"
        
        self._log_execution("grade_documents", {
            "question": question,
            "context": docs
        }, result)
        
        return result

    def _rewrite_node(self, state):
        """Rewrite query node logic"""
        messages = state["messages"]
        question = messages[-3].content

        msg = [
            HumanMessage(
                content=f""" \n 
            请仔细查看原始问题，尽量保持原问题的具体性和细节性，只在必要时进行细微调整。
            原始问题是：
            \n ------- \n
            {question} 
            \n ------- \n
            如果需要改写，请给出改进后的问题（注意保持具体性）：""",
            )
        ]

        response = self.llm.invoke(msg)
        return {"messages": [response]}

    def _generate_node(self, state):
        """Generate answer node logic"""
        messages = state["messages"]
        question = messages[-3].content
        docs = messages[-1].content

        prompt = ChatPromptTemplate.from_messages([
            ("system", LC_SYSTEM_PROMPT),
            ("human", """
                ---分析报告--- 
                请注意，下面提供的分析报告按**重要性降序排列**。
                
                {context}
                
                用户的问题是：
                {question}
                """),
        ])

        rag_chain = prompt | self.llm | StrOutputParser()
        response = rag_chain.invoke({
            "context": docs, 
            "question": question, 
            "response_type": response_type
        })
        
        self._log_execution("generate", {
            "question": question,
            "context": docs
        }, response)
        
        return {"messages": [AIMessage(content=response)]}

    def _reduce_node(self, state):
        """Reduce node logic for global search"""
        messages = state["messages"]
        question = messages[-3].content
        docs = messages[-1].content

        reduce_prompt = ChatPromptTemplate.from_messages([
            ("system", REDUCE_SYSTEM_PROMPT),
            ("human", """
                ---分析报告--- 
                {report_data}

                用户的问题是：
                {question}
                """),
        ])
        
        reduce_chain = reduce_prompt | self.llm | StrOutputParser()
        response = reduce_chain.invoke({
            "report_data": docs,
            "question": question,
            "response_type": response_type,
        })
        
        self._log_execution("reduce", {
            "question": question,
            "report_data": docs
        }, response)
        
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
        
        inputs = {"messages": [("user", query)]}
        for output in self.graph.stream(inputs, config=config):
            pprint.pprint(f"Output from node '{list(output.keys())[0]}':")
            pprint.pprint("---")
            pprint.pprint(output, indent=2, width=80, depth=None)
            pprint.pprint("\n---\n")
            
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
        
        inputs = {"messages": [("user", query)]}
        for output in self.graph.stream(inputs, config=config):
            pass
            
        chat_history = self.memory.get(config)["channel_values"]["messages"]
        return chat_history[-1].content

if __name__ == "__main__":
    agent = GraphAgent()
    
    # 调试模式
    result = agent.ask_with_trace("你好，想问一些问题。")
    print("Answer:", result["answer"])
    print("\nExecution trace:")
    pprint.pprint(result["execution_log"])
    
    queries = [
        "描述一下悟空第一次见到菩提祖师的场景？",
        "《悟空传》的主要人物有哪些？",
        "他们最后的结局是什么？"
    ]
    
    for query in queries:
        result = agent.ask_with_trace(query)
        print(f"\nQuestion: {query}")
        print("Answer:", result["answer"])
        print("\nExecution trace:")
        pprint.pprint(result["execution_log"])

    # 仅回答
    # print(agent.ask("你好，想问一些问题。"))
    # print(agent.ask("描述一下悟空第一次见到菩提祖师的场景？"))
    # print(agent.ask("《悟空传》的主要人物有哪些？"))
    # print(agent.ask("他们最后的结局是什么？"))