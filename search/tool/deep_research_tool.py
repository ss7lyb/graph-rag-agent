from typing import Dict, List, Any
import time
import re
import logging

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool

from search.tool.base import BaseSearchTool
from search.tool.hybrid_tool import HybridSearchTool
from search.tool.local_search_tool import LocalSearchTool
from search.tool.global_search_tool import GlobalSearchTool
from config.reasoning_prompts import BEGIN_SEARCH_QUERY, BEGIN_SEARCH_RESULT, END_SEARCH_RESULT, MAX_SEARCH_LIMIT, \
    END_SEARCH_QUERY, REASON_PROMPT, RELEVANT_EXTRACTION_PROMPT
from search.tool.reasoning.nlp import extract_between
from search.tool.reasoning.prompts import kb_prompt


class DeepResearchTool(BaseSearchTool):
    """
    深度研究工具：整合DeepResearcher功能到搜索工具架构中
    
    该工具实现了多步骤的搜索和推理过程，可以执行以下步骤：
    1. 思考分析用户问题
    2. 生成搜索查询
    3. 执行搜索
    4. 整合信息并进一步思考
    5. 迭代上述过程直到获得完整答案
    """
    
    def __init__(self):
        """初始化深度研究工具"""
        super().__init__(cache_dir="./cache/deep_research")
        
        self.hybrid_tool = HybridSearchTool()  # 用于关键词提取和混合搜索
        self.local_tool = LocalSearchTool()    # 用于KB检索
        self.global_tool = GlobalSearchTool()  # 用于社区检索
        
        # 初始化搜索函数
        self._kb_retrieve = self._create_kb_retrieval_func()
        self._kg_retrieve = self._create_kg_retrieval_func()
        
        # 设置最大迭代次数
        self.max_iterations = MAX_SEARCH_LIMIT
    
    def _setup_chains(self):
        # 我们主要依赖于其他工具的功能和思考方法
        pass
    
    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """从查询中提取关键词 """
        return self.hybrid_tool.extract_keywords(query)
    
    def _create_kb_retrieval_func(self):
        """创建知识库检索函数"""
        def kb_retrieve(question: str, limit: int = 5):
            try:
                # 使用 local_tool 的 search 方法进行检索
                result = self.local_tool.search(question)
                
                # 尝试从结果字符串中解析出数据字典
                import ast
                data_dict = {}
                data_match = re.search(r"({.*})", result)
                if data_match:
                    try:
                        data_dict = ast.literal_eval(data_match.group(1))
                    except Exception as parse_e:
                        logging.error(f"解析结果字符串失败: {parse_e}")
                
                # data_dict 可能直接包含数据，也可能包装在 "data" 键中
                if "data" in data_dict:
                    data = data_dict["data"]
                else:
                    data = data_dict
                
                # 从解析后的数据中提取各类信息，提供默认空列表
                entities = data.get("Entities", [])
                reports = data.get("Reports", [])
                relationships = data.get("Relationships", [])
                chunk_ids = data.get("Chunks", [])
                
                # 构建 chunks 列表（限制返回数量不超过 limit 个）
                chunks = []
                doc_aggs = []
                if chunk_ids:
                    for chunk_id in chunk_ids[:limit]:
                        chunks.append({
                            "chunk_id": chunk_id,
                            "text": f"Chunk内容: {chunk_id}",  # 实际应查询或缓存真实内容
                            "content_with_weight": f"Chunk内容: {chunk_id}",
                            "weight": 1.0,
                            "docnm_kwd": f"Document_{chunk_id}"
                        })
                        # 根据 chunk_id 构造文档 ID（取下划线前部分）
                        doc_id = chunk_id.split("_")[0] if "_" in chunk_id else chunk_id
                        if not any(d.get("doc_id") == doc_id for d in doc_aggs):
                            doc_aggs.append({
                                "doc_id": doc_id,
                                "title": f"Document: {doc_id}"
                            })
                
                return {
                    "chunks": chunks,
                    "doc_aggs": doc_aggs,
                    "entities": entities,
                    "reports": reports,
                    "relationships": relationships
                }
            except Exception as e:
                logging.error(f"知识库检索失败: {e}")
                return {
                    "chunks": [],
                    "doc_aggs": [],
                    "entities": [],
                    "reports": [],
                    "relationships": []
                }
        return kb_retrieve

    
    def _create_kg_retrieval_func(self):
        """创建知识图谱检索函数"""
        def kg_retrieve(question: str):
            """基于问题检索知识图谱内容"""
            try:
                # 使用global_tool获取社区信息
                results = self.global_tool.search(question)
                
                # 格式化结果为内容列表
                formatted_results = []
                
                if results and isinstance(results, list):
                    community_content = "## Relevant Knowledge Communities\n"
                    
                    for i, result in enumerate(results):
                        community_id = f"community_{i}"
                        community_content += f"### Community {community_id}\n"
                        community_content += f"Content: {result}\n\n"
                    
                    # 添加社区结果
                    formatted_results.append({
                        "chunk_id": "kg_community_result",
                        "content_with_weight": community_content,
                        "text": community_content,
                        "weight": 0.9,
                        "docnm_kwd": "Knowledge_Graph_Communities"
                    })
                
                return {"content_with_weight": formatted_results}
                
            except Exception as e:
                logging.error(f"知识图谱检索失败: {e}")
                return {"content_with_weight": []}
        
        return kg_retrieve
    
    def _remove_query_tags(self, text):
        """移除查询标签"""
        pattern = re.escape(BEGIN_SEARCH_QUERY) + r"(.*?)" + re.escape(END_SEARCH_QUERY)
        return re.sub(pattern, "", text, flags=re.DOTALL)
    
    def _remove_result_tags(self, text):
        """移除结果标签"""
        pattern = re.escape(BEGIN_SEARCH_RESULT) + r"(.*?)" + re.escape(END_SEARCH_RESULT)
        return re.sub(pattern, "", text, flags=re.DOTALL)

    def thinking(self, query: str):
        """
        执行深度研究推理过程
        
        参数:
            query: 用户问题
                
        返回:
            Dict: 包含思考过程和最终答案的字典
        """
        def rm_query_tags(line):
            pattern = re.escape(BEGIN_SEARCH_QUERY) + r"(.*?)" + re.escape(END_SEARCH_QUERY)
            return re.sub(pattern, "", line)

        def rm_result_tags(line):
            pattern = re.escape(BEGIN_SEARCH_RESULT) + r"(.*?)" + re.escape(END_SEARCH_RESULT)
            return re.sub(pattern, "", line)
        
        # 初始化结果容器
        chunk_info = {"chunks": [], "doc_aggs": []}
        executed_search_queries = []
        msg_history = []
        all_reasoning_steps = []
        all_retrieved_info = []  # 存储所有检索到的信息
        
        # 转换为正确的消息格式
        msg_history.append(HumanMessage(content=f'Question:\"{query}\"\n'))
        
        # 思考过程容器
        think = "<think>"
        
        # 迭代思考过程
        for iteration in range(self.max_iterations + 1):
            # 检查是否达到最大迭代次数
            if iteration == self.max_iterations - 1:
                summary_think = f"\n{BEGIN_SEARCH_RESULT}\n搜索次数已达上限。不允许继续搜索。\n{END_SEARCH_RESULT}\n"
                all_reasoning_steps.append(summary_think)
                msg_history.append(AIMessage(content=summary_think))
                think += rm_result_tags(summary_think)
                break

            # 添加用户消息，请求继续推理
            if isinstance(msg_history[-1], AIMessage):
                msg_history.append(HumanMessage(content="继续基于新信息进行推理分析。\n"))
            else:
                # 更新最后的用户消息
                last_content = msg_history[-1].content
                msg_history[-1] = HumanMessage(content=last_content + "\n\n继续基于新信息进行推理分析。\n")
            
            # 进行推理
            query_think = ""
            
            # 创建正确的消息格式
            formatted_messages = [SystemMessage(content=REASON_PROMPT)] + msg_history
            
            # 使用正确格式的消息调用LLM
            msg = self.llm.invoke(formatted_messages)
            
            if hasattr(msg, 'content'):
                query_think = msg.content
            else:
                query_think = str(msg)
            
            # 清理响应
            query_think = re.sub(r"<think>.*</think>", "", query_think, flags=re.DOTALL)
            if not query_think:
                continue
                
            # 更新思考过程
            think += rm_query_tags(query_think)
            all_reasoning_steps.append(query_think)
            
            # 提取搜索查询
            queries = extract_between(query_think, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)
            if not queries:
                if iteration > 0:
                    break  # 没有新的查询，结束迭代
                queries = [query]  # 使用原始问题作为查询
            
            # 处理每个搜索查询
            for search_query in queries:
                logging.info(f"[THINK]Query: {iteration}. {search_query}")
                msg_history.append(AIMessage(content=search_query))
                think += f"\n\n> {iteration + 1}. {search_query}\n\n"
                
                # 检查是否已执行过相同查询
                if search_query in executed_search_queries:
                    summary_think = f"\n{BEGIN_SEARCH_RESULT}\n已搜索过该查询。请参考前面的结果。\n{END_SEARCH_RESULT}\n"
                    all_reasoning_steps.append(summary_think)
                    msg_history.append(HumanMessage(content=summary_think))
                    think += rm_result_tags(summary_think)
                    continue
                
                # 记录已执行查询
                executed_search_queries.append(search_query)
                
                # 准备截断的推理历史，与原始实现保持一致
                truncated_prev_reasoning = ""
                for i, step in enumerate(all_reasoning_steps):
                    truncated_prev_reasoning += f"Step {i + 1}: {step}\n\n"

                prev_steps = truncated_prev_reasoning.split('\n\n')
                if len(prev_steps) <= 5:
                    truncated_prev_reasoning = '\n\n'.join(prev_steps)
                else:
                    truncated_prev_reasoning = ''
                    for i, step in enumerate(prev_steps):
                        if i == 0 or i >= len(prev_steps) - 4 or BEGIN_SEARCH_QUERY in step or BEGIN_SEARCH_RESULT in step:
                            truncated_prev_reasoning += step + '\n\n'
                        else:
                            if truncated_prev_reasoning[-len('\n\n...\n\n'):] != '\n\n...\n\n':
                                truncated_prev_reasoning += '...\n\n'
                truncated_prev_reasoning = truncated_prev_reasoning.strip('\n')
                
                # 检索知识
                kbinfos = self._kb_retrieve(search_query) if self._kb_retrieve else {"chunks": [], "doc_aggs": []}

                # 添加知识图谱结果
                if self._kg_retrieve:
                    ck = self._kg_retrieve(search_query)
                    if ck["content_with_weight"]:
                        kbinfos["chunks"].extend(ck["content_with_weight"])
                
                # 合并块信息用于引用
                if not chunk_info["chunks"]:
                    for k in chunk_info.keys():
                        chunk_info[k] = kbinfos[k]
                else:
                    cids = [c.get("chunk_id") for c in chunk_info["chunks"] if "chunk_id" in c]
                    for c in kbinfos["chunks"]:
                        if "chunk_id" in c and c["chunk_id"] in cids:
                            continue
                        chunk_info["chunks"].append(c)
                    dids = [d.get("doc_id") for d in chunk_info["doc_aggs"] if "doc_id" in d]
                    for d in kbinfos["doc_aggs"]:
                        if "doc_id" in d and d["doc_id"] in dids:
                            continue
                        chunk_info["doc_aggs"].append(d)
                
                # 生成摘要
                think += "\n\n"
                
                # 构建提取相关信息的提示
                extract_prompt = RELEVANT_EXTRACTION_PROMPT.format(
                    prev_reasoning=truncated_prev_reasoning,
                    search_query=search_query,
                    document="\n".join(kb_prompt(kbinfos, 4096))
                )
                
                # 使用正确的消息格式
                extraction_msg = self.llm.invoke([
                    SystemMessage(content=extract_prompt),
                    HumanMessage(content=f'基于当前的搜索查询"{search_query}"和前面的推理步骤，分析每个知识来源并找出有用信息。')
                ])
                
                if hasattr(extraction_msg, 'content'):
                    summary_think = extraction_msg.content
                else:
                    summary_think = str(extraction_msg)
                
                # 保存重要信息，用于最终生成
                if "**Final Information**" in summary_think and "No helpful information found" not in summary_think:
                    # 提取有用信息并存储
                    useful_info = summary_think.split("**Final Information**")[1].strip()
                    all_retrieved_info.append(useful_info)
                
                # 更新推理历史
                all_reasoning_steps.append(summary_think)
                msg_history.append(HumanMessage(content=f"\n\n{BEGIN_SEARCH_RESULT}{summary_think}{END_SEARCH_RESULT}\n\n"))
                think += rm_result_tags(summary_think)
                logging.info(f"[THINK]Summary: {iteration}. {summary_think}")
        
        # 完成思考过程
        think += "</think>"
        
        # 生成最终答案
        retrieved_content = "\n\n".join(all_retrieved_info)
        final_prompt = f"""
        基于以下思考过程和检索到的信息，回答用户问题。提供详细、准确、全面的回答，引用相关来源。

        用户问题: "{query}"

        检索到的信息:
        {retrieved_content}

        思考过程:
        {think}

        请生成综合性的最终回答，不需要解释你的思考过程，直接给出结论。
        """
        
        # 使用字符串形式传递最终提示
        answer = self.llm.invoke(final_prompt)
        content = answer.content if hasattr(answer, 'content') else answer
        
        # 返回结果
        result = {
            "thinking_process": think,
            "answer": content,
            "reference": chunk_info,
            "retrieved_info": all_retrieved_info,
        }
        
        return result

    def search(self, query_input: Any) -> str:
        """
        执行深度研究搜索
        
        参数:
            query_input: 搜索查询或包含查询的字典
            
        返回:
            str: 搜索结果
        """
        overall_start = time.time()
        
        # 解析输入
        if isinstance(query_input, dict) and "query" in query_input:
            query = query_input["query"]
        else:
            query = str(query_input)
        
        # 检查缓存
        cache_key = f"deep:{query}"
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            return cached_result
        
        try:
            result = self.thinking(query)
            answer = result["answer"]
            chunk_info = result.get("reference", {})
            
            # 格式化参考资料
            references = []
            if "doc_aggs" in chunk_info:
                for doc in chunk_info["doc_aggs"]:
                    references.append(doc.get("doc_id", ""))
            
            # 添加引用信息
            if references and "{'data': {'Chunks':" not in answer:
                ref_str = ", ".join([f"'{ref}'" for ref in references[:5]])
                answer += f"\n\n{{'data': {{'Chunks':[{ref_str}] }} }}"
            
            # 缓存结果
            self.cache_manager.set(cache_key, answer)
            
            # 记录总时间
            total_time = time.time() - overall_start
            self.performance_metrics["total_time"] = total_time
            
            return answer
            
        except Exception as e:
            error_msg = f"深度研究过程中出错: {str(e)}"
            print(error_msg)
            return error_msg
    
    def get_tool(self) -> BaseTool:
        """获取搜索工具"""
        class DeepResearchRetrievalTool(BaseTool):
            name : str = "deep_research"
            description : str = "深度研究工具：通过多轮推理和搜索解决复杂问题，尤其适用于需要深入分析的查询。"
            
            def _run(self_tool, query: Any) -> str:
                return self.search(query)
            
            def _arun(self_tool, query: Any) -> str:
                raise NotImplementedError("异步执行未实现")
        
        return DeepResearchRetrievalTool()
    
    def get_thinking_tool(self) -> BaseTool:
        """获取思考过程可见的工具版本"""
        class DeepThinkingTool(BaseTool):
            name : str = "deep_thinking"
            description : str = "深度思考工具：显示完整思考过程的深度研究，适用于需要查看推理步骤的情况。"
            
            def _run(self_tool, query: Any) -> Dict:
                # 解析输入
                if isinstance(query, dict) and "query" in query:
                    tk_query = query["query"]
                else:
                    tk_query = str(query)
                
                # 执行思考过程
                return self.thinking(tk_query)
            
            def _arun(self_tool, query: Any) -> Dict:
                raise NotImplementedError("异步执行未实现")
        
        return DeepThinkingTool()
    
    def close(self):
        """关闭资源"""
        # 调用父类方法
        super().close()
        
        # 关闭复用的工具资源
        if hasattr(self, 'naive_tool'):
            self.naive_tool.close()