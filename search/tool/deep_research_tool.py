from typing import Dict, List, Any, Optional
import time
import re
import logging
import json
import traceback
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from search.tool.base import BaseSearchTool
from search.tool.hybrid_tool import HybridSearchTool
from search.tool.local_search_tool import LocalSearchTool
from search.tool.global_search_tool import GlobalSearchTool
from config.reasoning_prompts import BEGIN_SEARCH_QUERY, BEGIN_SEARCH_RESULT, END_SEARCH_RESULT, MAX_SEARCH_LIMIT, \
    END_SEARCH_QUERY, RELEVANT_EXTRACTION_PROMPT, SUB_QUERY_PROMPT, FOLLOWUP_QUERY_PROMPT, FINAL_ANSWER_PROMPT
from search.tool.reasoning.nlp import extract_between
from search.tool.reasoning.prompts import kb_prompt
from search.tool.reasoning.thinking import ThinkingEngine
from search.tool.reasoning.validator import AnswerValidator
from search.tool.reasoning.search import DualPathSearcher, QueryGenerator
from config.settings import KB_NAME


class DeepResearchTool(BaseSearchTool):
    """
    深度研究工具：整合多种搜索策略，实现多步骤的思考-搜索-推理过程
    
    该工具实现了多步骤的研究过程，可以执行以下步骤：
    1. 思考分析用户问题
    2. 生成搜索查询
    3. 执行搜索
    4. 整合信息并进一步思考
    5. 迭代上述过程直到获得完整答案
    """
    
    def __init__(self):
        """初始化深度研究工具"""
        super().__init__(cache_dir="./cache/deep_research")
        
        # 初始化各种工具，用于不同阶段的搜索
        self.hybrid_tool = HybridSearchTool()  # 用于关键词提取和混合搜索
        self.global_tool = GlobalSearchTool()  # 用于社区检索
        self.local_tool = LocalSearchTool()    # 用于本地搜索
        
        # 初始化思考引擎
        self.thinking_engine = ThinkingEngine(self.llm)
        
        # 初始化查询生成器
        self.query_generator = QueryGenerator(
            self.llm, 
            SUB_QUERY_PROMPT, 
            FOLLOWUP_QUERY_PROMPT
        )
        
        # 初始化答案验证器
        self.validator = AnswerValidator(self.extract_keywords)
        
        # 初始化搜索器
        self._kb_retrieve = self._create_kb_retrieval_func()
        self._kg_retrieve = self._create_kg_retrieval_func()
        self.dual_searcher = DualPathSearcher(
            self._kb_retrieve, 
            self._kg_retrieve, 
            KB_NAME
        )
        
        # 存储重要信息
        self.all_retrieved_info = []
        
        # 设置最大迭代次数
        self.max_iterations = MAX_SEARCH_LIMIT
        
        # 用于存储执行日志
        self.execution_logs = []
    
    def _setup_chains(self):
        """设置处理链"""
        # 深度研究工具主要依赖于其他工具的功能和思考方法
        pass
    
    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """从查询中提取关键词"""
        return self.hybrid_tool.extract_keywords(query)
    
    def _parse_search_result(self, result):
        """
        解析搜索结果，支持多种格式
        
        参数:
            result: 搜索返回的原始结果
            
        返回:
            Dict: 解析后的结构化数据
        """
        # 已经是字典，直接返回
        if isinstance(result, dict):
            return result
        
        # 字符串结果需要解析
        if isinstance(result, str):
            # 尝试JSON解析
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                pass
            
            # 使用正则表达式提取JSON对象
            json_patterns = [
                r'{\s*"data"\s*:\s*(\{.*\})\s*}',  # {"data": {...}}
                r'(\{.*\})',                       # {...}
            ]
            
            for pattern in json_patterns:
                matches = re.search(pattern, result, re.DOTALL)
                if matches:
                    try:
                        import ast
                        extracted = matches.group(1)
                        parsed = ast.literal_eval(extracted)
                        return {"data": parsed}
                    except (SyntaxError, ValueError):
                        continue
            
            # 尝试提取Chunk IDs
            chunks_pattern = r'Chunks\s*:\s*\[(.*?)\]'
            chunks_match = re.search(chunks_pattern, result, re.DOTALL)
            if chunks_match:
                try:
                    chunk_text = chunks_match.group(1)
                    # 清理并分割
                    chunks = [c.strip("' \t\n\"") for c in chunk_text.split(",")]
                    chunks = [c for c in chunks if c]  # 移除空字符串
                    return {"data": {"Chunks": chunks}}
                except Exception:
                    pass
        
        # 无法解析，将整个内容作为文本
        return {"data": {"text": str(result)}}
    
    def _get_chunk_content(self, chunk_id: str) -> Optional[str]:
        """
        根据chunk_id获取真实内容
        
        参数:
            chunk_id: 文本块ID
            
        返回:
            str: 文本块内容，如果找不到则返回None
        """
        try:
            # 使用Neo4j查询获取chunk内容
            query = """
            MATCH (c:__Chunk__ {id: $chunk_id})
            RETURN c.text AS text
            """
            
            result = self.db_query(query, {"chunk_id": chunk_id})
            
            if not result.empty and 'text' in result.columns:
                return result.iloc[0]['text']
            return None
        except Exception as e:
            print(f"[获取Chunk内容] 错误: {str(e)}")
            return None
    
    def _create_kb_retrieval_func(self):
        """
        创建知识库检索函数
        
        返回:
            function: 知识库检索函数
        """
        def kb_retrieve(question: str, limit: int = 5):
            """基于问题检索知识库内容"""
            try:
                # 记录开始检索
                self._log(f"[KB检索] 开始搜索: {question}")

                # 使用本地搜索工具
                result = self.local_tool.search(question)
                self._log(f"[KB检索] 原始结果: {result}" if isinstance(result, str) else f"[KB检索] 原始结果类型: {type(result)}")
                
                # 检查结果是否为空
                if not result:
                    print("[KB检索] 搜索结果为空")
                    return {
                        "chunks": [],
                        "doc_aggs": [],
                        "entities": [],
                        "relationships": [],
                        "Chunks": []
                    }
                    
                # 解析结果
                try:
                    data_dict = self._parse_search_result(result)
                    self._log(f"[KB检索] 解析结果: {data_dict.keys()}")
                except Exception as parse_e:
                    print(f"[KB检索] 解析结果失败: {parse_e}")
                    # 如果解析失败但结果是字符串，创建一个简单的chunk
                    if isinstance(result, str) and len(result) > 10:
                        return {
                            "chunks": [{
                                "chunk_id": "text_content",
                                "text": result,
                                "content_with_weight": result,
                                "weight": 1.0
                            }],
                            "doc_aggs": [],
                            "entities": [],
                            "relationships": [],
                            "Chunks": ["text_content"]
                        }
                    return {
                        "chunks": [],
                        "doc_aggs": [],
                        "entities": [],
                        "relationships": [],
                        "Chunks": []
                    }
                
                # 标准化数据结构
                if "data" in data_dict:
                    data = data_dict["data"]
                else:
                    data = data_dict
                
                # 提取各类信息
                entities = data.get("Entities", [])
                reports = data.get("Reports", [])
                relationships = data.get("Relationships", [])
                chunk_ids = data.get("Chunks", [])
                
                # 如果data中已经有完整的chunks列表，直接使用
                if "chunks" in data and isinstance(data["chunks"], list) and data["chunks"]:
                    return data
                
                # 否则构建 chunks 列表
                chunks = []
                doc_aggs = []
                
                # 检查是否有真实的chunk_ids
                if chunk_ids:
                    for chunk_id in chunk_ids[:limit]:
                        # 尝试获取真实内容
                        chunk_content = self._get_chunk_content(chunk_id)
                        text = chunk_content or f"Chunk内容: {chunk_id}"
                        
                        chunks.append({
                            "chunk_id": chunk_id,
                            "text": text,
                            "content_with_weight": text,
                            "weight": 1.0,
                            "docnm_kwd": f"Document_{chunk_id}"
                        })
                        
                        # 构造文档聚合
                        doc_id = chunk_id.split("_")[0] if "_" in chunk_id else chunk_id
                        if not any(d.get("doc_id") == doc_id for d in doc_aggs):
                            doc_aggs.append({
                                "doc_id": doc_id,
                                "title": f"Document: {doc_id}"
                            })
                
                # 如果原始结果是字符串且没有找到chunks，将整个文本作为一个chunk
                elif isinstance(result, str) and len(result) > 10 and not chunks:
                    chunks.append({
                        "chunk_id": "text_result",
                        "text": result,
                        "content_with_weight": result,
                        "weight": 1.0,
                        "docnm_kwd": "Document_text"
                    })
                    doc_aggs.append({
                        "doc_id": "text",
                        "title": "Document: text"
                    })
                    chunk_ids = ["text_result"]
                
                # 记录结果统计
                self._log(f"[KB检索] 结果: {len(chunks)}个chunks, {len(entities)}个实体, {len(relationships)}个关系")
                
                return {
                    "chunks": chunks,
                    "doc_aggs": doc_aggs,
                    "entities": entities,
                    "reports": reports,
                    "relationships": relationships,
                    "Chunks": [c.get("chunk_id") for c in chunks]
                }
            except Exception as e:
                print(f"[KB检索错误] {str(e)}")
                print(traceback.format_exc())
                return {
                    "chunks": [],
                    "doc_aggs": [],
                    "entities": [],
                    "relationships": [],
                    "Chunks": []
                }
        
        return kb_retrieve
    
    def _create_kg_retrieval_func(self):
        """
        创建知识图谱检索函数
        
        返回:
            function: 知识图谱检索函数
        """
        def kg_retrieve(question: str):
            """基于问题检索知识图谱内容"""
            try:
                # 使用全局搜索工具获取社区信息
                results = self.global_tool.search(question)
                
                # 格式化结果为内容列表
                formatted_results = []
                
                if results and isinstance(results, list):
                    community_content = "## 相关知识社区\n"
                    
                    for i, result in enumerate(results):
                        community_id = f"community_{i}"
                        community_content += f"### 社区 {community_id}\n"
                        community_content += f"内容: {result}\n\n"
                    
                    # 添加社区结果
                    formatted_results.append({
                        "chunk_id": "kg_community_result",
                        "content_with_weight": community_content,
                        "text": community_content,
                        "weight": 0.9,
                        "docnm_kwd": "知识图谱社区"
                    })
                
                return {"content_with_weight": formatted_results}
                
            except Exception as e:
                logging.error(f"知识图谱检索失败: {e}")
                return {"content_with_weight": []}
        
        return kg_retrieve
    
    def _generate_final_answer(self, query: str, retrieved_content: str, thinking_process: str) -> str:
        """
        基于检索的信息和思考过程生成最终答案
        
        参数:
            query: 原始查询
            retrieved_content: 已检索的内容
            thinking_process: 思考过程
            
        返回:
            str: 最终答案，包含思考过程
        """
        try:
            # 调用LLM生成最终答案
            response = self.llm.invoke(FINAL_ANSWER_PROMPT.format(
                query=query,
                retrieved_content=retrieved_content,
                thinking_process=thinking_process
            ))
            
            answer = response.content if hasattr(response, 'content') else str(response)
            
            # 将思考过程添加到答案中，使用Markdown引用格式
            formatted_answer = f"<think>{thinking_process}</think>\n\n{answer}"
            
            return formatted_answer
        except Exception as e:
            print(f"[最终答案生成错误] {str(e)}")
            return f"生成最终答案时出错: {str(e)}"
        
    def _log(self, message):
        """记录执行日志"""
        self.execution_logs.append(message)
        print(message)  # 同时打印到控制台
    
    def thinking(self, query: str):
        """
        执行深度研究推理过程
        
        参数:
            query: 用户问题
                    
        返回:
            Dict: 包含思考过程和最终答案的字典
        """
        # 清空执行日志
        self.execution_logs = []
        self._log(f"[深度研究] 开始处理查询: {query}")
        
        # 初始化结果容器
        chunk_info = {"chunks": [], "doc_aggs": []}
        self.all_retrieved_info = []
        
        # 初始化思考引擎
        self.thinking_engine.initialize_with_query(query)

        think = ""
        
        # 迭代思考过程
        for iteration in range(self.max_iterations):
            self._log(f"[深度研究] 开始第{iteration + 1}轮迭代")
            
            # 检查是否达到最大迭代次数
            if iteration >= self.max_iterations - 1:
                summary_think = f"\n{BEGIN_SEARCH_RESULT}\n搜索次数已达上限。不允许继续搜索。\n{END_SEARCH_RESULT}\n"
                self.thinking_engine.add_reasoning_step(summary_think)
                self.thinking_engine.add_human_message(summary_think)
                think += self.thinking_engine.remove_result_tags(summary_think)
                break

            # 更新消息历史，请求继续推理
            self.thinking_engine.update_continue_message()
            
            # 生成下一个查询
            result = self.thinking_engine.generate_next_query()
            
            # 处理生成结果
            if result["status"] == "empty":
                self._log("[深度研究] 生成的思考内容为空")
                continue
            elif result["status"] == "error":
                self._log(f"[深度研究] 生成查询出错: {result.get('error', '未知错误')}")
                break
            elif result["status"] == "answer_ready":
                self._log("[深度研究] AI认为已有足够信息生成答案")
                break
                
            # 获取生成的思考内容
            query_think = result["content"]
            think += self.thinking_engine.remove_query_tags(query_think)
            
            # 获取搜索查询
            queries = result["queries"]
            
            # 如果没有生成搜索查询但不是第一轮，考虑结束
            if not queries and iteration > 0:
                if not self.all_retrieved_info:
                    # 如果还没有检索到任何信息，强制使用原始查询
                    queries = [query]
                    self._log("[深度研究] 没有检索到信息，使用原始查询")
                else:
                    # 已有信息，结束迭代
                    self._log("[深度研究] 没有生成新查询且已有信息，结束迭代")
                    break
            
            # 处理每个搜索查询
            for search_query in queries:
                self._log(f"[深度研究] 执行查询: {search_query}")
                
                # 检查是否已执行过相同查询
                if self.thinking_engine.has_executed_query(search_query):
                    summary_think = f"\n{BEGIN_SEARCH_RESULT}\n已搜索过该查询。请参考前面的结果。\n{END_SEARCH_RESULT}\n"
                    self.thinking_engine.add_reasoning_step(summary_think)
                    self.thinking_engine.add_human_message(summary_think)
                    think += self.thinking_engine.remove_result_tags(summary_think)
                    continue
                
                # 记录已执行查询
                self.thinking_engine.add_executed_query(search_query)
                
                # 将搜索查询添加到消息历史
                self.thinking_engine.add_ai_message(f"{BEGIN_SEARCH_QUERY}{search_query}{END_SEARCH_QUERY}")
                think += f"\n\n> {iteration + 1}. {search_query}\n\n"
                
                # 执行实际搜索
                kbinfos = self.dual_searcher.search(search_query)
                
                # 检查搜索结果是否为空
                has_results = (
                    kbinfos.get("chunks", []) or 
                    kbinfos.get("entities", []) or 
                    kbinfos.get("relationships", [])
                )
                
                if not has_results:
                    no_result_msg = f"\n{BEGIN_SEARCH_RESULT}\n没有找到与'{search_query}'相关的信息。请尝试使用不同的关键词进行搜索。\n{END_SEARCH_RESULT}\n"
                    self.thinking_engine.add_reasoning_step(no_result_msg)
                    self.thinking_engine.add_human_message(no_result_msg)
                    think += self.thinking_engine.remove_result_tags(no_result_msg)
                    continue
                
                # 正常处理有结果的情况
                truncated_prev_reasoning = self.thinking_engine.prepare_truncated_reasoning()
                
                # 合并块信息
                chunk_info = self.dual_searcher._merge_results(chunk_info, kbinfos)
                
                # 构建提取相关信息的提示
                kb_prompt_result = "\n".join(kb_prompt(kbinfos, 4096))
                extract_prompt = RELEVANT_EXTRACTION_PROMPT.format(
                    prev_reasoning=truncated_prev_reasoning,
                    search_query=search_query,
                    document=kb_prompt_result
                )
                
                # 使用LLM提取有用信息
                extraction_msg = self.llm.invoke([
                    SystemMessage(content=extract_prompt),
                    HumanMessage(content=f'基于当前的搜索查询"{search_query}"和前面的推理步骤，分析每个知识来源并找出有用信息。')
                ])
                
                summary_think = extraction_msg.content if hasattr(extraction_msg, 'content') else str(extraction_msg)
                
                # 保存重要信息
                has_useful_info = (
                    "**Final Information**" in summary_think and 
                    "No helpful information found" not in summary_think
                )
                
                if has_useful_info:
                    useful_info = summary_think.split("**Final Information**")[1].strip()
                    self.all_retrieved_info.append(useful_info)
                    self._log(f"[深度研究] 发现有用信息: {useful_info}")
                else:
                    self._log("[深度研究] 未发现有用信息")
                
                # 更新推理历史
                self.thinking_engine.add_reasoning_step(summary_think)
                self.thinking_engine.add_human_message(f"\n{BEGIN_SEARCH_RESULT}{summary_think}{END_SEARCH_RESULT}\n")
                think += self.thinking_engine.remove_result_tags(summary_think)
        
        # 生成最终答案
        # 确保至少执行了一次搜索
        if not self.thinking_engine.executed_search_queries:
            return {
                "thinking_process": think,
                "answer": f"抱歉，我无法回答关于'{query}'的问题，因为没有找到相关信息。",
                "reference": chunk_info,
                "retrieved_info": [],
                "execution_logs": self.execution_logs,
            }
        
        # 使用检索到的信息生成答案
        retrieved_content = "\n\n".join(self.all_retrieved_info)
        final_answer = self._generate_final_answer(query, retrieved_content, think)
        
        # 返回结果
        result = {
            "thinking_process": think,
            "answer": final_answer,
            "reference": chunk_info,
            "retrieved_info": self.all_retrieved_info,
            "execution_logs": self.execution_logs,
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
        
        # 记录开始搜索
        self._log(f"[深度搜索] 开始处理查询...")
        
        # 解析输入
        if isinstance(query_input, dict) and "query" in query_input:
            query = query_input["query"]
        else:
            query = str(query_input)
        
        self._log(f"[深度搜索] 解析后的查询: {query}")
        
        # 检查缓存
        cache_key = f"deep:{query}"
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            self._log(f"[深度搜索] 缓存命中，返回缓存结果")
            return cached_result
        
        try:
            # 执行思考过程
            self._log(f"[深度搜索] 开始执行思考过程")
            result = self.thinking(query)
            answer = result["answer"]
            chunk_info = result.get("reference", {})
            
            # 格式化参考资料
            references = []
            if "doc_aggs" in chunk_info:
                for doc in chunk_info["doc_aggs"]:
                    doc_id = doc.get("doc_id", "")
                    if doc_id and doc_id not in references:
                        references.append(doc_id)
            
            # 添加引用信息
            if references and "{'data': {'Chunks':" not in answer:
                ref_str = ", ".join([f"'{ref}'" for ref in references[:5]])
                answer += f"\n\n{{'data': {{'Chunks':[{ref_str}] }} }}"
            
            # 验证答案质量
            validation_results = self.validator.validate(query, answer)
            if validation_results["passed"]:
                self._log(f"[深度搜索] 答案验证通过，缓存结果")
                self.cache_manager.set(cache_key, answer)
            else:
                self._log(f"[深度搜索] 答案验证失败，不缓存")
            
            # 记录总时间
            total_time = time.time() - overall_start
            self._log(f"[深度搜索] 完成，耗时 {total_time:.2f}秒")
            self.performance_metrics["total_time"] = total_time
            
            return answer
                
        except Exception as e:
            error_msg = f"深度研究过程中出错: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return f"抱歉，处理您的问题时遇到了错误: {str(e)}"
    
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
        if hasattr(self, 'hybrid_tool'):
            self.hybrid_tool.close()
        if hasattr(self, 'global_tool'):
            self.global_tool.close()
        if hasattr(self, 'local_tool'):
            self.local_tool.close()