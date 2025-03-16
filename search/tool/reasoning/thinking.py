import re
from typing import List, Dict, Any
import logging
import traceback
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from search.tool.reasoning.nlp import extract_between
from config.reasoning_prompts import BEGIN_SEARCH_QUERY, BEGIN_SEARCH_RESULT, END_SEARCH_RESULT, REASON_PROMPT, END_SEARCH_QUERY


class ThinkingEngine:
    """
    思考引擎类：负责管理多轮迭代的思考过程
    提供思考历史管理和转换功能
    """
    
    def __init__(self, llm):
        """
        初始化思考引擎
        
        参数:
            llm: 大语言模型实例，用于生成思考内容
        """
        self.llm = llm
        self.all_reasoning_steps = []
        self.msg_history = []
        self.executed_search_queries = []
    
    def initialize_with_query(self, query: str):
        """
        使用初始查询初始化思考历史
        
        参数:
            query: 用户问题
        """
        self.all_reasoning_steps = []
        self.msg_history = [HumanMessage(content=f'问题:"{query}"\n')]
        self.executed_search_queries = []
    
    def add_reasoning_step(self, content: str):
        """
        添加推理步骤
        
        参数:
            content: 步骤内容
        """
        self.all_reasoning_steps.append(content)
    
    def remove_query_tags(self, text: str) -> str:
        """
        移除文本中的查询标签
        
        参数:
            text: 包含标签的文本
            
        返回:
            str: 移除标签后的文本
        """
        pattern = re.escape(BEGIN_SEARCH_QUERY) + r"(.*?)" + re.escape(END_SEARCH_QUERY)
        return re.sub(pattern, "", text, flags=re.DOTALL)
    
    def remove_result_tags(self, text: str) -> str:
        """
        移除文本中的结果标签
        
        参数:
            text: 包含标签的文本
            
        返回:
            str: 移除标签后的文本
        """
        pattern = re.escape(BEGIN_SEARCH_RESULT) + r"(.*?)" + re.escape(END_SEARCH_RESULT)
        return re.sub(pattern, "", text, flags=re.DOTALL)
    
    def extract_queries(self, text: str) -> List[str]:
        """
        从文本中提取搜索查询
        
        参数:
            text: 包含查询的文本
            
        返回:
            List[str]: 提取的查询列表
        """
        return extract_between(text, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)
    
    def generate_next_query(self) -> Dict[str, Any]:
        """
        生成下一步搜索查询
        
        返回:
            Dict: 包含查询和状态信息的字典
        """
        # 使用LLM进行推理分析，获取下一个搜索查询
        formatted_messages = [SystemMessage(content=REASON_PROMPT)] + self.msg_history
        
        try:
            # 调用LLM生成查询
            msg = self.llm.invoke(formatted_messages)
            query_think = msg.content if hasattr(msg, 'content') else str(msg)
            
            # 清理响应
            query_think = re.sub(r"<think>.*</think>", "", query_think, flags=re.DOTALL)
            if not query_think:
                return {"status": "empty", "content": None, "queries": []}
                
            # 更新思考过程
            clean_think = self.remove_query_tags(query_think)
            self.add_reasoning_step(query_think)
            
            # 从AI响应中提取搜索查询
            queries = self.extract_queries(query_think)
            
            # 如果没有生成搜索查询，检查是否应该结束
            if not queries:
                # 检查是否包含最终答案标记
                if "**回答**" in query_think or "足够的信息" in query_think:
                    return {
                        "status": "answer_ready", 
                        "content": query_think,
                        "queries": []
                    }
                
                # 没有明确结束标志，就继续
                return {
                    "status": "no_query", 
                    "content": query_think,
                    "queries": []
                }
            
            # 有查询，继续搜索
            return {
                "status": "has_query", 
                "content": query_think,
                "queries": queries
            }
            
        except Exception as e:
            error_msg = f"生成查询时出错: {str(e)}\n{traceback.format_exc()}"
            logging.error(error_msg)
            return {"status": "error", "error": error_msg, "queries": []}
    
    def add_ai_message(self, content: str):
        """
        添加AI消息到历史记录
        
        参数:
            content: 消息内容
        """
        self.msg_history.append(AIMessage(content=content))
    
    def add_human_message(self, content: str):
        """
        添加用户消息到历史记录
        
        参数:
            content: 消息内容
        """
        self.msg_history.append(HumanMessage(content=content))
    
    def update_continue_message(self):
        """更新最后的消息，请求继续推理"""
        if len(self.msg_history) > 0 and isinstance(self.msg_history[-1], AIMessage):
            self.add_human_message("继续基于新信息进行推理分析。\n")
        elif len(self.msg_history) > 0:
            # 更新最后的用户消息
            last_content = self.msg_history[-1].content
            self.msg_history[-1] = HumanMessage(content=last_content + "\n\n继续基于新信息进行推理分析。\n")
    
    def prepare_truncated_reasoning(self) -> str:
        """
        准备截断的推理历史，保留关键部分以减少token使用
        
        返回:
            str: 截断的推理历史
        """
        all_reasoning_steps = self.all_reasoning_steps
        
        if not all_reasoning_steps:
            return ""
            
        # 如果步骤少于5个，保留全部
        if len(all_reasoning_steps) <= 5:
            steps_text = ""
            for i, step in enumerate(all_reasoning_steps):
                steps_text += f"Step {i + 1}: {step}\n\n"
            return steps_text.strip()
        
        # 否则，保留第一步、最后4步和包含查询/结果的步骤
        important_steps = []
        
        # 总是包含第一步
        important_steps.append((0, all_reasoning_steps[0]))
        
        # 包含最后4步
        for i in range(max(1, len(all_reasoning_steps) - 4), len(all_reasoning_steps)):
            important_steps.append((i, all_reasoning_steps[i]))
        
        # 包含中间包含搜索查询或结果的步骤
        for i in range(1, len(all_reasoning_steps) - 4):
            step = all_reasoning_steps[i]
            if BEGIN_SEARCH_QUERY in step or BEGIN_SEARCH_RESULT in step:
                important_steps.append((i, step))
        
        # 按原始顺序排序
        important_steps.sort(key=lambda x: x[0])
        
        # 格式化结果
        truncated = ""
        prev_idx = -1
        
        for idx, step in important_steps:
            # 如果有间隔，添加省略号
            if idx > prev_idx + 1:
                truncated += "...\n\n"
            
            truncated += f"Step {idx + 1}: {step}\n\n"
            prev_idx = idx
        
        return truncated.strip()
    
    def get_full_thinking(self) -> str:
        """
        获取完整的思考过程
        
        返回:
            str: 格式化的思考过程
        """
        thinking = "<think>\n"
        
        for step in self.all_reasoning_steps:
            clean_step = self.remove_query_tags(step)
            clean_step = self.remove_result_tags(clean_step)
            thinking += clean_step + "\n\n"
            
        thinking += "</think>"
        return thinking
    
    def has_executed_query(self, query: str) -> bool:
        """
        检查是否已经执行过相同的查询
        
        参数:
            query: 查询字符串
            
        返回:
            bool: 是否已执行过
        """
        return query in self.executed_search_queries
    
    def add_executed_query(self, query: str):
        """
        添加已执行的查询
        
        参数:
            query: 已执行的查询字符串
        """
        self.executed_search_queries.append(query)