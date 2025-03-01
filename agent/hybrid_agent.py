# 使用 hybrid search tool 的agent
from typing import Annotated, Sequence, TypedDict, List, Dict, Any
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
from config.prompt import LC_SYSTEM_PROMPT
from config.settings import response_type
from search.tool.hybrid_tool import HybridSearchTool

class QueryCache:
    """简单的查询缓存实现"""
    
    def __init__(self, max_size=100, cache_dir="./cache/hybrid_agent"):
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
        
        # 初始化增强型搜索工具
        self.search_tool = HybridSearchTool()
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
        
        # 注意：移除了agent级别的缓存，避免过早拦截工具调用
        # agent节点需要执行工具调用，因此不应缓存
        
        # 使用工具处理请求
        model = self.llm.bind_tools(self.tools)
        response = model.invoke(messages)
        
        self._log_execution("agent", messages, response)
        return {"messages": [response]}

    def _extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """提取查询关键词"""
        # 使用安全的缓存键
        safe_query = query.strip()
        cache_key = f"keywords_{self._get_safe_cache_key(safe_query)}"
        
        # 检查缓存
        cached_keywords = self.query_cache.get(cache_key)
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
            self.query_cache.set(cache_key, keywords)
            
            return keywords
        except Exception as e:
            print(f"关键词提取失败: {e}")
            # 出错时返回默认空关键词
            return {"low_level": [], "high_level": []}

    def _generate_node(self, state):
        """Generate answer node logic"""
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

        # 使用安全的缓存键检查缓存
        safe_question = question.strip()
        cache_key = f"generate_{self._get_safe_cache_key(safe_question)}"
        cached_result = self.query_cache.get(cache_key)
        if cached_result:
            self._log_execution("generate", 
                               {"question": safe_question, "docs_length": len(docs)}, 
                               "缓存命中")
            return {"messages": [AIMessage(content=cached_result)]}

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
        try:
            response = rag_chain.invoke({
                "context": docs, 
                "question": question, 
                "response_type": response_type
            })
            
            # 只缓存有效的生成结果
            if response and len(response) > 10:
                self.query_cache.set(cache_key, response)
            
            self._log_execution("generate", 
                              {"question": safe_question, "docs_length": len(docs)}, 
                              response)
            
            return {"messages": [AIMessage(content=response)]}
        except Exception as e:
            error_msg = f"生成回答时出错: {str(e)}"
            self._log_execution("generate_error", 
                              {"question": safe_question, "docs_length": len(docs)}, 
                              error_msg)
            return {"messages": [AIMessage(content=f"抱歉，我无法回答这个问题。技术原因: {str(e)}")]}
        

    def _get_safe_cache_key(self, query):
        """生成安全的缓存键，处理中文和特殊字符"""
        # 使用更安全的方式生成缓存键
        return hashlib.md5(query.encode('utf-8')).hexdigest()
        
    def ask_with_trace(self, query: str, thread_id: str = "default", recursion_limit: int = 5) -> Dict:
        """Ask a question and get both the answer and execution trace"""
        self.execution_log = []  # Reset execution log
        
        # 检查是否有完整问答的缓存，使用更安全的缓存键方案
        safe_query = query.strip()
        cached_answer = self.query_cache.get(f"full_qa_{self._get_safe_cache_key(safe_query)}")
        if cached_answer:
            print(f"完整问答缓存命中: {safe_query[:30]}...")
            return {
                "answer": cached_answer,
                "execution_log": [{"node": "cache_hit", "timestamp": time.time(), "input": safe_query, "output": "缓存命中"}]
            }
        
        config = {
            "configurable": {
                "thread_id": thread_id,
                "recursion_limit": recursion_limit
            }
        }
        
        inputs = {"messages": [HumanMessage(content=query)]}
        try:
            for output in self.graph.stream(inputs, config=config):
                pprint.pprint(f"Output from node '{list(output.keys())[0]}':")
                pprint.pprint("---")
                pprint.pprint(output, indent=2, width=80, depth=None)
                pprint.pprint("\n---\n")
                
            chat_history = self.memory.get(config)["channel_values"]["messages"]
            answer = chat_history[-1].content
            
            # 只在成功获取答案时缓存结果
            if answer and len(answer) > 10:  # 确保答案有实际内容
                self.query_cache.set(f"full_qa_{self._get_safe_cache_key(safe_query)}", answer)
            
            return {
                "answer": answer,
                "execution_log": self.execution_log
            }
        except Exception as e:
            print(f"处理查询时出错: {e}")
            return {
                "answer": f"抱歉，处理您的问题时遇到了错误。请稍后再试或换一种提问方式。错误详情: {str(e)}",
                "execution_log": self.execution_log + [{"node": "error", "timestamp": time.time(), "input": query, "output": str(e)}]
            }

    def ask(self, query: str, thread_id: str = "default", recursion_limit: int = 5):
        """Ask a question to the agent"""
        # 检查是否有完整问答的缓存，使用更安全的缓存键方案
        safe_query = query.strip()
        cached_answer = self.query_cache.get(f"full_qa_{self._get_safe_cache_key(safe_query)}")
        if cached_answer:
            print(f"完整问答缓存命中: {safe_query[:30]}...")
            return cached_answer
            
        config = {
            "configurable": {
                "thread_id": thread_id,
                "recursion_limit": recursion_limit
            }
        }
        
        inputs = {"messages": [HumanMessage(content=query)]}
        try:
            # 使用带异常处理的方式运行图
            for output in self.graph.stream(inputs, config=config):
                pass
                
            chat_history = self.memory.get(config)["channel_values"]["messages"]
            answer = chat_history[-1].content
            
            # 只在成功获取答案时缓存结果
            if answer and len(answer) > 10:  # 确保答案有实际内容
                self.query_cache.set(f"full_qa_{self._get_safe_cache_key(safe_query)}", answer)
            
            return answer
        except Exception as e:
            print(f"处理查询时出错: {e}")
            return f"抱歉，处理您的问题时遇到了错误。请稍后再试或换一种提问方式。错误详情: {str(e)}"
        
    def close(self):
        """关闭资源"""
        if self.search_tool:
            self.search_tool.close()

if __name__ == "__main__":
    agent = GraphAgent()
    
    try:
        # 测试基本问答
        print(agent.ask("你好，想问一些问题。"))
        print(agent.ask("《悟空传》的主要人物有哪些？"))
        print(agent.ask("孙悟空跟女妖之间有什么故事？"))
        print(agent.ask("他最后的选择是什么？"))
    finally:
        # 确保资源被正确关闭
        agent.close()