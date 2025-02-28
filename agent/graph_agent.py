from typing import Annotated, Literal, Sequence, TypedDict, List, Dict, Any
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
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
from config.prompt import LC_SYSTEM_PROMPT, REDUCE_SYSTEM_PROMPT
from config.settings import response_type
from search.tool.local_search_tool import LocalSearchTool
from search.tool.global_search_tool import GlobalSearchTool

class QueryCache:
    """简单的查询缓存实现"""
    
    def __init__(self, max_size=100, cache_dir="./cache/graph_agent"):
        self.cache = {}
        self.max_size = max_size
        self.cache_dir = cache_dir
        
        # 确保缓存目录存在
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        # 从磁盘加载缓存
        self._load_from_disk()
    
    def _get_cache_key(self, query):
        """生成缓存键"""
        return hashlib.md5(query.encode()).hexdigest()
    
    def _get_cache_path(self, key):
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f"{key}.json")
    
    def _load_from_disk(self):
        """从磁盘加载缓存"""
        if os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                if filename.endswith(".json"):
                    key = filename[:-5]  # 移除 .json 后缀
                    try:
                        with open(os.path.join(self.cache_dir, filename), 'r', encoding='utf-8') as f:
                            self.cache[key] = json.load(f)
                    except Exception as e:
                        print(f"加载缓存文件 {filename} 时出错: {e}")
    
    def get(self, query):
        """获取缓存的结果"""
        key = self._get_cache_key(query)
        result = self.cache.get(key)
        if result:
            print(f"缓存命中: {query[:30]}...")
            return result
        return None
    
    def set(self, query, result):
        """设置缓存结果"""
        key = self._get_cache_key(query)
        
        # 如果缓存已满，删除最旧的项
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            
            # 删除磁盘上的缓存文件
            cache_path = self._get_cache_path(oldest_key)
            if os.path.exists(cache_path):
                os.remove(cache_path)
        
        # 更新内存缓存
        self.cache[key] = result
        
        # 写入磁盘缓存
        try:
            with open(self._get_cache_path(key), 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"写入缓存文件时出错: {e}")

class GraphAgent:
    def __init__(self):
        self.llm = get_llm_model()
        self.memory = MemorySaver()
        self.execution_log = []
        self.query_cache = QueryCache()
        
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
            {
                "generate": "generate", 
                "reduce": "reduce"
            }
        )
        workflow.add_edge("generate", END)
        workflow.add_edge("reduce", END)
        
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
        # 检查缓存
        cached_keywords = self.query_cache.get(f"keywords_{query}")
        if cached_keywords:
            return cached_keywords
            
        # 使用LLM提取关键词
        try:
            # 使用简单的prompt模板，避免复杂格式
            prompt = f"""提取以下查询的关键词:
            查询: {query}
            
            请提取两类关键词:
            1. 低级关键词: 具体实体、名称、术语
            2. 高级关键词: 主题、概念、领域
            
            以JSON格式返回。
            """
            
            result = self.llm.invoke(prompt)
            
            # 解析LLM返回的内容
            content = result.content if hasattr(result, 'content') else result
            
            # 尝试提取JSON部分
            import re
            json_match = re.search(r'({.*})', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                try:
                    keywords = json.loads(json_str)
                    # 确保结果有正确的格式
                    if not isinstance(keywords, dict):
                        keywords = {}
                    if "low_level" not in keywords:
                        keywords["low_level"] = []
                    if "high_level" not in keywords:
                        keywords["high_level"] = []
                        
                    # 缓存结果
                    self.query_cache.set(f"keywords_{query}", keywords)
                    return keywords
                except:
                    pass
        except Exception as e:
            print(f"关键词提取失败: {e}")
            
        # 如果提取失败，返回默认值
        default_keywords = {"low_level": [], "high_level": []}
        return default_keywords

    def _grade_documents(self, state) -> Literal["generate", "reduce"]:
        """Grade documents relevance"""
        messages = state["messages"]
        retrieve_message = messages[-2]
        
        # 检查是否为全局检索工具调用
        tool_calls = retrieve_message.additional_kwargs.get("tool_calls", [])
        if tool_calls and tool_calls[0].get("function", {}).get("name") == "global_retriever":
            self._log_execution("grade_documents", messages, "reduce")
            return "reduce"

        # 简化的相关性评分 - 总是返回 "generate"，避免 "rewrite" 状态
        question = messages[-3].content
        docs = messages[-1].content
        
        # 从问题中提取关键词
        keywords = []
        if hasattr(messages[-3], 'additional_kwargs') and messages[-3].additional_kwargs:
            kw_data = messages[-3].additional_kwargs.get("keywords", {})
            if isinstance(kw_data, dict):
                keywords = kw_data.get("low_level", []) + kw_data.get("high_level", [])
        
        if not keywords:
            # 如果没有提取到关键词，使用简单的关键词提取
            keywords = [word for word in question.lower().split() if len(word) > 2]
        
        # 计算关键词匹配率
        docs_text = docs.lower()
        matches = sum(1 for keyword in keywords if keyword.lower() in docs_text)
        match_rate = matches / len(keywords) if keywords else 0
        
        # 始终返回 "generate" 而不是 "rewrite"，避免路由错误
        result = "generate"
        
        self._log_execution("grade_documents", {
            "question": question,
            "keywords": keywords,
            "match_rate": match_rate
        }, result)
        
        return result

    def _generate_node(self, state):
        """Generate answer node logic"""
        messages = state["messages"]
        question = messages[-3].content
        docs = messages[-1].content

        # Check cache for generation results
        cached_result = self.query_cache.get(f"generate_{question}")
        if cached_result:
            self._log_execution("generate", 
                               {"question": question, "docs_length": len(docs)}, 
                               cached_result)
            return {"messages": [AIMessage(content=cached_result)]}

        prompt = ChatPromptTemplate.from_messages([
        ("system", LC_SYSTEM_PROMPT),
        ("human", """
            ---分析报告--- 
            请注意，下面提供的分析报告按**重要性降序排列**。
            
            {context}
            
            用户的问题是：
            {question}
            
            请严格按照以下格式输出回答：
            1. 使用三级标题(###)标记主题
            2. 主要内容用清晰的段落展示
            3. 最后必须用"#### 引用数据"标记引用部分，列出用到的数据来源
            """),
        ])

        rag_chain = prompt | self.llm | StrOutputParser()
        response = rag_chain.invoke({
            "context": docs, 
            "question": question, 
            "response_type": response_type
        })
        
        # Save to cache
        self.query_cache.set(f"generate_{question}", response)
        
        self._log_execution("generate", 
                           {"question": question, "docs_length": len(docs)}, 
                           response)
        
        return {"messages": [AIMessage(content=response)]}

    def _reduce_node(self, state):
        """Reduce node logic for global search"""
        messages = state["messages"]
        question = messages[-3].content
        docs = messages[-1].content

        # Check cache for reduce results
        cached_result = self.query_cache.get(f"reduce_{question}")
        if cached_result:
            self._log_execution("reduce", 
                               {"question": question, "docs_length": len(docs)}, 
                               cached_result)
            return {"messages": [AIMessage(content=cached_result)]}

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
        
        # Save to cache
        self.query_cache.set(f"reduce_{question}", response)
        
        self._log_execution("reduce", 
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
        
        inputs = {"messages": [HumanMessage(content=query)]}
        for output in self.graph.stream(inputs, config=config):
            pass
            
        chat_history = self.memory.get(config)["channel_values"]["messages"]
        return chat_history[-1].content

if __name__ == "__main__":
    agent = GraphAgent()
    
    # 仅回答
    print(agent.ask("你好，想问一些问题。"))
    print(agent.ask("《悟空传》的主要人物有哪些？"))
    print(agent.ask("孙悟空跟女妖之间有什么故事？"))
    print(agent.ask("他最后的选择是什么？"))
    