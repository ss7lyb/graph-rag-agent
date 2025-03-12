from typing import Dict, List, Any, Optional
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
    END_SEARCH_QUERY, REASON_PROMPT, RELEVANT_EXTRACTION_PROMPT, SUB_QUERY_PROMPT, FOLLOWUP_QUERY_PROMPT, FINAL_ANSWER_PROMPT
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
        self.global_tool = GlobalSearchTool()  # 用于社区检索
        self.local_tool = LocalSearchTool()
        
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
                # 记录开始检索
                print(f"[KB检索] 开始搜索: {question}")

                result = self.local_tool.search(question)
                print(f"[KB检索] 原始结果: {result[:200]}..." if isinstance(result, str) else f"[KB检索] 原始结果类型: {type(result)}")
                
                # 检查结果是否为空字符串或None
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
                    print(f"[KB检索] 解析结果: {data_dict.keys()}")
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
                print(f"[KB检索] 结果: {len(chunks)}个chunks, {len(entities)}个实体, {len(relationships)}个关系")
                
                return {
                    "chunks": chunks,
                    "doc_aggs": doc_aggs,
                    "entities": entities,
                    "reports": reports,
                    "relationships": relationships,
                    "Chunks": [c.get("chunk_id") for c in chunks]
                }
            except Exception as e:
                import traceback
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
                import json
                return json.loads(result)
            except json.JSONDecodeError:
                pass
            
            # 使用正则表达式提取JSON对象
            import re
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
    
    def _generate_sub_queries(self, original_query: str) -> List[str]:
        """
        将原始查询分解为多个子查询
        
        参数:
            original_query: 原始用户查询
            
        返回:
            List[str]: 子查询列表
        """
        try:
            # 调用LLM生成子查询
            response = self.llm.invoke(SUB_QUERY_PROMPT.format(original_query=original_query))
            content = response.content if hasattr(response, 'content') else str(response)
            
            # 提取列表文本
            import re
            list_text = re.search(r'\[.*\]', content, re.DOTALL)
            if list_text:
                try:
                    # 解析列表
                    import ast
                    sub_queries = ast.literal_eval(list_text.group(0))
                    return sub_queries
                except Exception as e:
                    print(f"[子查询生成] 解析列表失败: {str(e)}")
            
            # 如果无法解析，返回原始查询
            return [original_query]
        except Exception as e:
            print(f"[子查询生成错误] {str(e)}")
            return [original_query]
        
    def _generate_followup_queries(self, original_query: str, retrieved_info: List[str]) -> List[str]:
        """
        基于已检索的信息生成跟进查询
        
        参数:
            original_query: 原始查询
            retrieved_info: 已检索的信息列表
            
        返回:
            List[str]: 跟进查询列表，如果不需要则为空列表
        """
        # 如果没有检索到任何信息，或信息不足，返回空列表
        if not retrieved_info or len(retrieved_info) < 2:
            return []
        
        try:
            # 合并已检索信息（但限制长度）
            info_text = "\n\n".join(retrieved_info[-3:])  # 只使用最近的3条信息
            
            # 调用LLM生成跟进查询
            response = self.llm.invoke(FOLLOWUP_QUERY_PROMPT.format(
                original_query=original_query,
                retrieved_info=info_text
            ))
            content = response.content if hasattr(response, 'content') else str(response)
            
            # 提取列表文本
            import re
            list_text = re.search(r'\[.*\]', content, re.DOTALL)
            if list_text:
                try:
                    # 解析列表
                    import ast
                    followup_queries = ast.literal_eval(list_text.group(0))
                    
                    # 确保没有重复查询
                    unique_queries = []
                    for q in followup_queries:
                        if q not in unique_queries:
                            unique_queries.append(q)
                    
                    return unique_queries
                except Exception as e:
                    print(f"[跟进查询生成] 解析列表失败: {str(e)}")
            
            # 如果无法解析，返回空列表
            return []
        except Exception as e:
            print(f"[跟进查询生成错误] {str(e)}")
            return []
    

    def _generate_final_answer(self, query: str, retrieved_content: str, thinking_process: str) -> str:
        """
        基于检索的信息和思考过程生成最终答案
        
        参数:
            query: 原始查询
            retrieved_content: 已检索的内容
            thinking_process: 思考过程
            
        返回:
            str: 最终答案
        """
        try:
            # 调用LLM生成最终答案
            response = self.llm.invoke(FINAL_ANSWER_PROMPT.format(
                query=query,
                retrieved_content=retrieved_content,
                thinking_process=thinking_process
            ))
            
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            print(f"[最终答案生成错误] {str(e)}")
            return f"生成最终答案时出错: {str(e)}"
    

    def _prepare_truncated_reasoning(self, all_reasoning_steps: List[str]) -> str:
        """
        准备截断的推理历史，保留关键部分以减少token使用
        
        参数:
            all_reasoning_steps: 所有推理步骤
            
        返回:
            str: 截断的推理历史
        """
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

    def _merge_chunks(self, current_info: Dict, new_info: Dict) -> Dict:
        """
        合并块信息，避免重复
        
        参数:
            current_info: 当前块信息
            new_info: 新块信息
            
        返回:
            Dict: 合并后的块信息
        """
        # 初始化结果字典
        result = {
            "chunks": current_info.get("chunks", []).copy(),
            "doc_aggs": current_info.get("doc_aggs", []).copy()
        }
        
        # 如果当前没有chunks，直接使用新信息
        if not result["chunks"]:
            return new_info
        
        # 已存在的chunk_id和doc_id集合
        existing_chunk_ids = set(c.get("chunk_id") for c in result["chunks"] if "chunk_id" in c)
        existing_doc_ids = set(d.get("doc_id") for d in result["doc_aggs"] if "doc_id" in d)
        
        # 合并chunks，避免重复
        for chunk in new_info.get("chunks", []):
            chunk_id = chunk.get("chunk_id")
            # 只添加不存在的chunks
            if chunk_id and chunk_id not in existing_chunk_ids:
                result["chunks"].append(chunk)
                existing_chunk_ids.add(chunk_id)
            elif not chunk_id:
                # 如果没有chunk_id，使用内容作为唯一性判断
                content = chunk.get("text", "")
                if not any(c.get("text") == content for c in result["chunks"]):
                    result["chunks"].append(chunk)
        
        # 合并doc_aggs，避免重复
        for doc in new_info.get("doc_aggs", []):
            doc_id = doc.get("doc_id")
            if doc_id and doc_id not in existing_doc_ids:
                result["doc_aggs"].append(doc)
                existing_doc_ids.add(doc_id)
        
        # 复制其他字段
        for key in new_info:
            if key not in ["chunks", "doc_aggs"]:
                if key not in result:
                    result[key] = new_info[key]
                elif isinstance(result[key], list) and isinstance(new_info[key], list):
                    # 合并列表类型的字段
                    result[key].extend([item for item in new_info[key] if item not in result[key]])
        
        return result
    
    def _validate_answer(self, query: str, answer: str) -> bool:
        """
        验证生成答案的质量
        
        参数:
            query: 原始查询
            answer: 生成的答案
            
        返回:
            bool: 答案是否满足质量要求
        """
        # 检查最小长度
        if len(answer) < 50:
            print(f"[验证] 答案太短: {len(answer)}字符")
            return False
            
        # 检查错误消息
        error_patterns = [
            "抱歉，处理您的问题时遇到了错误",
            "技术原因:",
            "无法获取",
            "无法回答这个问题",
            "没有找到相关信息",
            "对不起，我不能"
        ]
        
        for pattern in error_patterns:
            if pattern in answer:
                print(f"[验证] 答案包含错误模式: {pattern}")
                return False
                
        # 相关性检查 - 检查问题关键词是否在答案中出现
        keywords = self.extract_keywords(query)
        if keywords:
            high_level_keywords = keywords.get("high_level", [])
            low_level_keywords = keywords.get("low_level", [])
            
            # 至少有一个高级关键词应该在答案中出现
            if high_level_keywords:
                keyword_found = any(keyword.lower() in answer.lower() for keyword in high_level_keywords)
                if not keyword_found:
                    print(f"[验证] 答案未包含任何高级关键词: {high_level_keywords}")
                    return False
                    
            # 至少有一半的低级关键词应该在答案中出现
            if low_level_keywords and len(low_level_keywords) > 1:
                matches = sum(1 for keyword in low_level_keywords if keyword.lower() in answer.lower())
                if matches < len(low_level_keywords) / 2:
                    print(f"[验证] 答案未包含足够的低级关键词: {matches}/{len(low_level_keywords)}")
                    return False
        
        print("[验证] 答案通过验证")
        return True

    def thinking(self, query: str):
        """
        执行深度研究推理过程
        
        参数:
            query: 用户问题
                    
        返回:
            Dict: 包含思考过程和最终答案的字典
        """
        print(f"[深度研究] 开始处理查询: {query}")
        
        # 1. 初始化结果容器
        chunk_info = {"chunks": [], "doc_aggs": []}
        executed_search_queries = []
        msg_history = []
        all_reasoning_steps = []
        all_retrieved_info = []
        
        # 创建初始消息
        msg_history.append(HumanMessage(content=f'Question:\"{query}\"\n'))
        
        # 思考过程容器
        think = "<think>"
        
        # 迭代思考过程
        for iteration in range(self.max_iterations):
            print(f"[深度研究] 开始第{iteration + 1}轮迭代")
            
            # 检查是否达到最大迭代次数
            if iteration >= self.max_iterations - 1:
                summary_think = f"\n{BEGIN_SEARCH_RESULT}\n搜索次数已达上限。不允许继续搜索。\n{END_SEARCH_RESULT}\n"
                all_reasoning_steps.append(summary_think)
                msg_history.append(HumanMessage(content=summary_think))
                think += self._remove_result_tags(summary_think)
                break

            # 添加用户消息，请求继续推理
            if len(msg_history) > 0 and isinstance(msg_history[-1], AIMessage):
                msg_history.append(HumanMessage(content="继续基于新信息进行推理分析。\n"))
            elif len(msg_history) > 0:
                # 更新最后的用户消息
                last_content = msg_history[-1].content
                msg_history[-1] = HumanMessage(content=last_content + "\n\n继续基于新信息进行推理分析。\n")
            
            # 使用LLM进行推理分析，获取下一个搜索查询
            formatted_messages = [SystemMessage(content=REASON_PROMPT)] + msg_history
            
            try:
                # 调用LLM生成查询
                msg = self.llm.invoke(formatted_messages)
                query_think = msg.content if hasattr(msg, 'content') else str(msg)
                
                # 清理响应
                query_think = re.sub(r"<think>.*</think>", "", query_think, flags=re.DOTALL)
                if not query_think:
                    continue
                    
                # 更新思考过程
                think += self._remove_query_tags(query_think)
                all_reasoning_steps.append(query_think)
                
                # 从AI响应中提取搜索查询
                queries = extract_between(query_think, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)
                
                # 如果没有生成搜索查询，检查是否应该结束
                if not queries:
                    # 检查是否包含最终答案标记
                    if "**回答**" in query_think or "足够的信息" in query_think:
                        print("[深度研究] AI准备生成最终答案，但未执行搜索")
                        warning_msg = f"\n{BEGIN_SEARCH_RESULT}\n警告: 尚未完成足够的搜索。请先执行搜索，再得出结论。\n{END_SEARCH_RESULT}\n"
                        all_reasoning_steps.append(warning_msg)
                        msg_history.append(HumanMessage(content=warning_msg))
                        think += self._remove_result_tags(warning_msg)
                        continue
                    
                    # 如果是首次迭代但没有搜索查询，使用原始查询
                    if iteration == 0:
                        queries = [query]
                    else:
                        # 后续迭代没有搜索查询，结束迭代
                        break
                
                # 处理每个搜索查询
                for search_query in queries:
                    print(f"[深度研究] 执行查询: {search_query}")
                    
                    # 检查是否已执行过相同查询
                    if search_query in executed_search_queries:
                        summary_think = f"\n{BEGIN_SEARCH_RESULT}\n已搜索过该查询。请参考前面的结果。\n{END_SEARCH_RESULT}\n"
                        all_reasoning_steps.append(summary_think)
                        msg_history.append(HumanMessage(content=summary_think))
                        think += self._remove_result_tags(summary_think)
                        continue
                    
                    # 记录已执行查询
                    executed_search_queries.append(search_query)
                    
                    # 重要：将搜索查询添加到消息历史，确保AI知道它发出了查询
                    msg_history.append(AIMessage(content=f"{BEGIN_SEARCH_QUERY}{search_query}{END_SEARCH_QUERY}"))
                    think += f"\n\n> {iteration + 1}. {search_query}\n\n"
                    
                    # 执行实际搜索
                    kbinfos = self._dual_path_search(search_query)
                    
                    # 检查搜索结果是否为空
                    has_results = (
                        kbinfos.get("chunks", []) or 
                        kbinfos.get("entities", []) or 
                        kbinfos.get("relationships", [])
                    )
                    
                    if not has_results:
                        no_result_msg = f"\n{BEGIN_SEARCH_RESULT}\n没有找到与'{search_query}'相关的信息。请尝试使用不同的关键词进行搜索。\n{END_SEARCH_RESULT}\n"
                        all_reasoning_steps.append(no_result_msg)
                        msg_history.append(HumanMessage(content=no_result_msg))
                        think += self._remove_result_tags(no_result_msg)
                        continue
                    
                    # 正常处理有结果的情况
                    truncated_prev_reasoning = self._prepare_truncated_reasoning(all_reasoning_steps)
                    
                    # 合并块信息
                    chunk_info = self._merge_chunks(chunk_info, kbinfos)
                    
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
                        all_retrieved_info.append(useful_info)
                        print(f"[深度研究] 发现有用信息: {useful_info[:100]}...")
                    else:
                        print("[深度研究] 未发现有用信息")
                    
                    # 更新推理历史
                    all_reasoning_steps.append(summary_think)
                    msg_history.append(HumanMessage(content=f"\n{BEGIN_SEARCH_RESULT}{summary_think}{END_SEARCH_RESULT}\n"))
                    think += self._remove_result_tags(summary_think)
            
            except Exception as e:
                import traceback
                error_msg = f"[深度研究] 处理中出错: {str(e)}\n{traceback.format_exc()}"
                print(error_msg)
                break
        
        # 完成思考过程
        think += "</think>"
        
        # 生成最终答案
        # 确保至少执行了一次搜索
        if not executed_search_queries:
            return {
                "thinking_process": think,
                "answer": f"抱歉，我无法回答关于'{query}'的问题，因为没有找到相关信息。",
                "reference": chunk_info,
                "retrieved_info": []
            }
        
        # 使用检索到的信息生成答案
        retrieved_content = "\n\n".join(all_retrieved_info)
        final_answer = self._generate_final_answer(query, retrieved_content, think)
        
        # 返回结果
        result = {
            "thinking_process": think,
            "answer": final_answer,
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
        
        # 记录开始搜索
        print(f"[深度搜索] 开始处理查询...")
        
        # 解析输入
        if isinstance(query_input, dict) and "query" in query_input:
            query = query_input["query"]
        else:
            query = str(query_input)
        
        print(f"[深度搜索] 解析后的查询: {query}")
        
        # 检查缓存
        cache_key = f"deep:{query}"
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            print(f"[深度搜索] 缓存命中，返回缓存结果")
            return cached_result
        
        try:
            # 执行思考过程
            print(f"[深度搜索] 开始执行思考过程")
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
            if self._validate_answer(query, answer):
                print(f"[深度搜索] 答案验证通过，缓存结果")
                self.cache_manager.set(cache_key, answer)
            else:
                print(f"[深度搜索] 答案验证失败，不缓存")
            
            # 记录总时间
            total_time = time.time() - overall_start
            print(f"[深度搜索] 完成，耗时 {total_time:.2f}秒")
            self.performance_metrics["total_time"] = total_time
            
            return answer
                
        except Exception as e:
            import traceback
            error_msg = f"深度研究过程中出错: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return f"抱歉，处理您的问题时遇到了错误: {str(e)}"
        
    def _dual_path_search(self, query: str) -> Dict:
        """
        实现双路径搜索，同时尝试精确查询和带知识库名称的查询
        
        参数:
            query: 原始查询
            
        返回:
            Dict: 合并后的检索结果
        """
        from config.settings import KB_NAME
        
        # 精确查询
        precise_query = query.replace(KB_NAME, "").strip()
        # 带名称的查询
        kb_query = f"{KB_NAME} {query}" if KB_NAME.lower() not in query.lower() else query
        
        # 执行两种查询
        precise_results = self._kb_retrieve(precise_query)
        kb_results = self._kb_retrieve(kb_query)
        
        # 检查哪个结果更好（有更多chunks或entities）
        precise_content_count = len(precise_results.get("chunks", [])) + len(precise_results.get("entities", []))
        kb_content_count = len(kb_results.get("chunks", [])) + len(kb_results.get("entities", []))
        
        # 选择更好的结果，或合并两者
        if precise_content_count > 0 and precise_content_count >= kb_content_count:
            print(f"[双路径搜索] 精确查询结果更好: {precise_content_count} vs {kb_content_count}")
            return precise_results
        elif kb_content_count > 0:
            print(f"[双路径搜索] 带知识库名查询结果更好: {kb_content_count} vs {precise_content_count}")
            return kb_results
        else:
            # 合并结果
            print("[双路径搜索] 合并两种查询结果")
            return self._merge_chunks(precise_results, kb_results)
    
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