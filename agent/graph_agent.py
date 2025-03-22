from typing import List, Dict
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END
from langgraph.prebuilt import tools_condition
import asyncio

import json
import re

from config.prompt import LC_SYSTEM_PROMPT, REDUCE_SYSTEM_PROMPT
from config.settings import response_type
from search.tool.local_search_tool import LocalSearchTool
from search.tool.global_search_tool import GlobalSearchTool

from agent.base import BaseAgent


class GraphAgent(BaseAgent):
    """使用图结构的Agent实现"""
    
    def __init__(self):
        # 初始化本地和全局搜索工具
        self.local_tool = LocalSearchTool()
        self.global_tool = GlobalSearchTool()
        
        # 设置缓存目录
        self.cache_dir = "./cache/graph_agent"
        
        # 调用父类构造函数
        super().__init__(cache_dir=self.cache_dir)

    def _setup_tools(self) -> List:
        """设置工具"""
        return [
            self.local_tool.get_tool(),
            self.global_tool.search,
        ]
    
    def _add_retrieval_edges(self, workflow):
        """添加从检索到生成的边"""
        # 添加 reduce 节点
        workflow.add_node("reduce", self._reduce_node)
        
        # 添加条件边，根据文档评分决定路由
        workflow.add_conditional_edges(
            "retrieve",
            self._grade_documents,
            {
                "generate": "generate", 
                "reduce": "reduce"
            }
        )
        
        # 添加从 reduce 到结束的边
        workflow.add_edge("reduce", END)

    def _extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """提取查询关键词"""
        # 检查缓存
        cached_keywords = self.cache_manager.get(f"keywords:{query}")
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
                    self.cache_manager.set(f"keywords:{query}", keywords)
                    return keywords
                except:
                    pass
        except Exception as e:
            print(f"关键词提取失败: {e}")
            
        # 如果提取失败，返回默认值
        default_keywords = {"low_level": [], "high_level": []}
        return default_keywords

    def _grade_documents(self, state) -> str:
        """评估文档相关性 - 返回 'generate' 或 'reduce'"""
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
        """生成回答节点逻辑"""
        messages = state["messages"]
        question = messages[-3].content
        docs = messages[-1].content

        # 检查缓存
        cached_result = self.cache_manager.get(f"generate:{question}")
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
        
        # 缓存结果
        self.cache_manager.set(f"generate:{question}", response)
        
        self._log_execution("generate", 
                           {"question": question, "docs_length": len(docs)}, 
                           response)
        
        return {"messages": [AIMessage(content=response)]}

    def _reduce_node(self, state):
        """处理全局搜索的Reduce节点逻辑"""
        messages = state["messages"]
        question = messages[-3].content
        docs = messages[-1].content

        # 检查缓存
        cached_result = self.cache_manager.get(f"reduce:{question}")
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
        
        # 缓存结果
        self.cache_manager.set(f"reduce:{question}", response)
        
        self._log_execution("reduce", 
                           {"question": question, "docs_length": len(docs)}, 
                           response)
        
        return {"messages": [AIMessage(content=response)]}
    
    async def _generate_node_stream(self, state):
        """生成回答节点逻辑的流式版本"""
        messages = state["messages"]
        question = messages[-3].content
        docs = messages[-1].content

        # 检查缓存
        thread_id = state.get("configurable", {}).get("thread_id", "default")
        cached_result = self.cache_manager.get(f"generate:{question}", thread_id=thread_id)
        if cached_result:
            yield cached_result
            return

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

        # 使用流式模型
        rag_chain = prompt | self.stream_llm
        
        # 处理流式响应
        result = ""
        async for chunk in rag_chain.astream({
            "context": docs, 
            "question": question, 
            "response_type": response_type
        }):
            # 从 chunk 中提取内容
            if hasattr(chunk, "content"):
                token = chunk.content
            else:
                token = str(chunk)
                
            # 返回 token 给调用者
            yield token
            
            # 添加到完整结果
            result += token
        
        # 缓存完整结果
        if result and len(result) > 10:
            self.cache_manager.set(f"generate:{question}", result, thread_id=thread_id)
    
    async def _stream_process(self, inputs, config):
        """实现流式处理过程"""
        # 获取会话信息
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        query = inputs["messages"][-1].content
        
        # 检查缓存先
        cached_response = self.cache_manager.get(query.strip(), thread_id=thread_id)
        if cached_response:
            # 对于缓存的响应，分块返回
            chunk_size = 4
            for i in range(0, len(cached_response), chunk_size):
                yield cached_response[i:i+chunk_size]
                await asyncio.sleep(0.01)
            return
        
        # 处理工作流
        workflow_state = {"messages": [HumanMessage(content=query)]}
        
        # 执行 agent 节点
        agent_output = self._agent_node(workflow_state)
        workflow_state = {"messages": workflow_state["messages"] + agent_output["messages"]}
        
        # 检查是否需要使用工具
        tool_decision = tools_condition(workflow_state)
        if tool_decision == "tools":
            # 执行检索节点
            retrieve_output = await self._retrieve_node_async(workflow_state)
            workflow_state = {"messages": workflow_state["messages"] + retrieve_output["messages"]}
            
            # 流式生成节点输出
            async for token in self._generate_node_stream(workflow_state):
                yield token
        else:
            # 不需要工具，直接返回代理的响应
            final_msg = workflow_state["messages"][-1]
            content = final_msg.content if hasattr(final_msg, "content") else str(final_msg)
            
            # 分块返回
            chunk_size = 4
            for i in range(0, len(content), chunk_size):
                yield content[i:i+chunk_size]
                await asyncio.sleep(0.01)
    
    # 异步检索节点辅助方法
    async def _retrieve_node_async(self, state):
        """检索节点的异步版本，用于流式处理"""
        try:
            # 获取工具调用信息
            tool_calls = state["messages"][-1].additional_kwargs.get("tool_calls", [])
            if not tool_calls:
                # 尝试其他方式获取查询
                if hasattr(state["messages"][-1], "tool_calls") and state["messages"][-1].tool_calls:
                    tool_calls = state["messages"][-1].tool_calls
                    
            # 如果找到工具调用
            if tool_calls and len(tool_calls) > 0:
                # 获取第一个工具调用的信息
                tool_call = tool_calls[0]
                query = tool_call.get("args", {}).get("query", "")
                tool_call_id = tool_call.get("id", "tool_call_0")  # 提供默认ID
                
                # 执行搜索
                tool_result = self.local_tool.search(query)
                
                # 创建带有完整属性的工具消息
                return {
                    "messages": [
                        ToolMessage(
                            content=tool_result,
                            tool_call_id=tool_call_id,
                            name=tool_call.get("name", "search_tool")
                        )
                    ]
                }
            else:
                # 无法找到工具调用信息，返回错误
                return {
                    "messages": [
                        AIMessage(content="无法获取查询信息，请重试。")
                    ]
                }
        except Exception as e:
            # 处理错误
            error_msg = f"处理工具调用时出错: {str(e)}"
            print(error_msg)
            return {
                "messages": [
                    AIMessage(content=error_msg)
                ]
            }
        