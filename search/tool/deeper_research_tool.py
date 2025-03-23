from typing import Dict,Any, List, AsyncGenerator
import json
import time
import traceback
import asyncio
import re

from model.get_models import get_llm_model, get_embeddings_model
from graph.core import connection_manager
from config.prompt import (
    system_template_build_graph,
    human_template_build_graph
)
from config.settings import (
    entity_types,
    relationship_types
)
from config.reasoning_prompts import RELEVANT_EXTRACTION_PROMPT
from search.tool.reasoning.prompts import kb_prompt
from graph.extraction.entity_extractor import EntityRelationExtractor
from search.tool.deep_research_tool import DeepResearchTool
from search.tool.reasoning.community_enhance import CommunityAwareSearchEnhancer
from search.tool.reasoning.kg_builder import DynamicKnowledgeGraphBuilder
from search.tool.reasoning.evidence import EvidenceChainTracker

class DeeperResearchTool:
    """
    增强版深度研究工具
    
    整合社区感知、动态知识图谱和证据链跟踪等功能，
    提供更全面的深度研究能力
    """
    
    def __init__(self, 
                config=None, 
                llm=None, 
                embeddings=None,
                graph=None):
        """
        初始化增强版深度研究工具
        
        Args:
            config: 配置参数
            llm: 语言模型
            embeddings: 嵌入模型
            graph: 图数据库连接
        """
        # 初始化基础组件
        self.llm = llm or get_llm_model()
        self.embeddings = embeddings or get_embeddings_model()
        self.graph = graph or connection_manager.get_connection()
        
        # 初始化增强模块
        # 1. 社区感知搜索增强器
        self.community_search = CommunityAwareSearchEnhancer(
            self.graph, 
            self.embeddings, 
            self.llm
        )
        
        # 2. 动态知识图谱构建器
        self.knowledge_builder = DynamicKnowledgeGraphBuilder(
            self.graph,
            EntityRelationExtractor(self.llm, system_template_build_graph, human_template_build_graph, entity_types, relationship_types),
        )
        
        # 3. 证据链跟踪器
        self.evidence_tracker = EvidenceChainTracker()
        
        # 4. 继承原有的深度研究工具功能
        self.deep_research = DeepResearchTool()
        
        # 缓存设置
        self.enable_cache = True
        self.cache_dir = "./cache/deeper_research"
        
        # 确保缓存目录存在
        import os
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
        # 添加执行日志容器
        self.execution_logs = []
        
        # 添加性能指标跟踪
        self.performance_metrics = {"total_time": 0}
    
    def _log(self, message):
        """记录执行日志"""
        self.execution_logs.append(message)
        print(message)  # 同时打印到控制台
    
    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """从查询中提取关键词"""
        return self.deep_research.extract_keywords(query)
    
    def thinking(self, query: str) -> Dict[str, Any]:
        """
        执行增强版深度研究推理过程
        
        Args:
            query: 用户问题
                    
        Returns:
            Dict: 包含思考过程和最终答案的字典
        """
        # 清空执行日志
        self.execution_logs = []
        self._log(f"\n[深度研究] 开始处理查询: {query}")
        
        # 提取关键词
        keywords = self.deep_research.extract_keywords(query)
        self._log(f"\n[深度研究] 提取关键词: {keywords}")
        
        # 开始新的查询跟踪
        query_id = self.evidence_tracker.start_new_query(query, keywords)
        
        # 步骤1: 社区感知增强
        self._log(f"\n[深度研究] 开始社区感知分析")
        community_context = self.community_search.enhance_search(query, keywords)
        
        # 步骤2: 使用社区信息增强搜索策略
        search_strategy = community_context.get("search_strategy", {})
        follow_up_queries = search_strategy.get("follow_up_queries", [])
        
        if follow_up_queries:
            self._log(f"\n[深度研究] 社区分析生成的后续查询: {follow_up_queries}")
        
        # 添加增强搜索策略的推理步骤
        strategy_step_id = self.evidence_tracker.add_reasoning_step(
            query_id, 
            "knowledge_community_analysis", 
            f"基于社区分析，识别了关键实体和相关查询策略。将探索以下后续查询: {follow_up_queries}"
        )
        
        # 记录社区信息作为证据
        community_summaries = community_context.get("community_info", {}).get("summaries", [])
        if community_summaries:
            self._log(f"\n[深度研究] 找到 {len(community_summaries)} 个相关社区")
            for i, summary in enumerate(community_summaries):
                self._log(f"\n[深度研究] 社区 {i+1} 摘要: {summary[:100]}...")
                self.evidence_tracker.add_evidence(
                    strategy_step_id,
                    f"community_summary_{i}",
                    summary,
                    "community_knowledge"
                )
        
        # 步骤3: 构建初始知识图谱
        initial_entities = search_strategy.get("focus_entities", [])
        if initial_entities:
            self._log(f"\n[深度研究] 构建知识图谱，关注实体: {initial_entities}")
            self.knowledge_builder.build_query_graph(query, initial_entities, depth=1)
            
            # 获取核心实体
            central_entities = self.knowledge_builder.get_central_entities(limit=5)
            central_entity_ids = [e["id"] for e in central_entities]
            
            # 添加知识图谱分析步骤
            kg_step_id = self.evidence_tracker.add_reasoning_step(
                query_id,
                "knowledge_graph_analysis",
                f"构建了初始知识图谱，识别出核心实体: {central_entity_ids}"
            )
            
            # 记录知识图谱信息作为证据
            kg_info = {
                "entity_count": self.knowledge_builder.knowledge_graph.number_of_nodes(),
                "relation_count": self.knowledge_builder.knowledge_graph.number_of_edges(),
                "central_entities": central_entity_ids
            }
            self._log(f"\n[深度研究] 知识图谱统计: {kg_info['entity_count']} 个实体, {kg_info['relation_count']} 个关系")
            
            self.evidence_tracker.add_evidence(
                kg_step_id,
                "knowledge_graph",
                json.dumps(kg_info),
                "graph_structure"
            )
        
        # 步骤4: 深度研究思考过程        
        # 初始化结果容器
        think = ""
        self.deep_research.all_retrieved_info = []
        
        # 初始化思考引擎
        self._log(f"\n[深度研究] 初始化思考引擎")
        self.deep_research.thinking_engine.initialize_with_query(query)
        
        # 添加已有的社区信息和知识图谱信息作为上下文
        context_message = ""
        if community_summaries:
            context_message += "找到了相关的知识社区信息:\n"
            context_message += "\n".join(community_summaries)
            
        # 如果有跟进查询，添加到思考过程
        if follow_up_queries:
            context_message += "\n\n可能的深入探索方向:\n"
            context_message += "\n".join([f"- {q}" for q in follow_up_queries])
        
        # 将上下文添加到思考引擎
        if context_message:
            self.deep_research.thinking_engine.add_reasoning_step(context_message)
            think += context_message + "\n\n"
        
        # 迭代思考过程
        for iteration in range(self.deep_research.max_iterations):
            self._log(f"\n[深度研究] 开始第{iteration + 1}轮迭代\n")
            
            # 跟踪迭代步骤
            iteration_step_id = self.evidence_tracker.add_reasoning_step(
                query_id,
                f"iteration_{iteration}",
                f"\n开始第 {iteration + 1} 轮迭代思考\n"
            )
            
            # 检查是否达到最大迭代次数
            if iteration >= self.deep_research.max_iterations - 1:
                summary_think = f"\n搜索次数已达上限。不允许继续搜索。\n"
                self.deep_research.thinking_engine.add_reasoning_step(summary_think)
                self.deep_research.thinking_engine.add_human_message(summary_think)
                think += self.deep_research.thinking_engine.remove_result_tags(summary_think)
                self._log("\n[深度研究] 达到最大迭代次数，结束搜索")
                break

            # 更新消息历史，请求继续推理
            self.deep_research.thinking_engine.update_continue_message()
            
            # 生成下一个查询
            result = self.deep_research.thinking_engine.generate_next_query()
            
            # 处理生成结果
            if result["status"] == "empty":
                self._log("\n[深度研究] 生成的思考内容为空")
                self.evidence_tracker.add_evidence(
                    iteration_step_id,
                    "generation_result",
                    "生成的思考内容为空",
                    "reasoning_status"
                )
                continue
            elif result["status"] == "error":
                self._log(f"\n[深度研究] 生成查询出错: {result.get('error', '未知错误')}")
                self.evidence_tracker.add_evidence(
                    iteration_step_id,
                    "generation_error",
                    f"生成查询出错: {result.get('error', '未知错误')}",
                    "reasoning_status"
                )
                break
            elif result["status"] == "answer_ready":
                self._log("\n[深度研究] AI认为已有足够信息生成答案")
                self.evidence_tracker.add_evidence(
                    iteration_step_id,
                    "reasoning_complete",
                    "AI认为已有足够信息生成答案",
                    "reasoning_status"
                )
                break
                
            # 获取生成的思考内容
            query_think = result["content"]
            think += self.deep_research.thinking_engine.remove_query_tags(query_think)
            
            # 获取搜索查询
            queries = result["queries"]
            
            # 如果没有生成搜索查询但不是第一轮，考虑结束
            if not queries and iteration > 0:
                if not self.deep_research.all_retrieved_info:
                    # 如果还没有检索到任何信息，强制使用原始查询
                    queries = [query]
                    self._log("\n[深度研究] 没有检索到信息，使用原始查询")
                    self.evidence_tracker.add_evidence(
                        iteration_step_id,
                        "query_fallback",
                        "没有检索到信息，使用原始查询",
                        "reasoning_status"
                    )
                else:
                    # 已有信息，结束迭代
                    self._log("\n[深度研究] 没有生成新查询且已有信息，结束迭代")
                    self.evidence_tracker.add_evidence(
                        iteration_step_id,
                        "iteration_complete",
                        "没有生成新查询且已有信息，结束迭代",
                        "reasoning_status"
                    )
                    break
            
            # 处理每个搜索查询
            for search_query in queries:
                self._log(f"\n[深度研究] 执行查询: {search_query}")
                
                # 记录查询步骤
                search_step_id = self.evidence_tracker.add_reasoning_step(
                    query_id,
                    search_query,
                    f"执行查询: {search_query}"
                )
                
                # 检查是否已执行过相同查询
                if self.deep_research.thinking_engine.has_executed_query(search_query):
                    self._log(f"\n[深度研究] 已搜索过该查询: {search_query}")
                    summary_think = f"\n已搜索过该查询。请参考前面的结果。\n"
                    self.deep_research.thinking_engine.add_reasoning_step(summary_think)
                    self.deep_research.thinking_engine.add_human_message(summary_think)
                    think += self.deep_research.thinking_engine.remove_result_tags(summary_think)
                    
                    self.evidence_tracker.add_evidence(
                        search_step_id,
                        "duplicate_query",
                        "已搜索过该查询，跳过重复执行",
                        "search_status"
                    )
                    continue
                
                # 记录已执行查询
                self.deep_research.thinking_engine.add_executed_query(search_query)
                
                # 将搜索查询添加到消息历史
                self.deep_research.thinking_engine.add_ai_message(f"{search_query}")
                think += f"\n\n> {iteration + 1}. {search_query}\n\n"
                
                # 执行实际搜索
                self._log(f"\n[KB检索] 开始搜索: {search_query}")
                kbinfos = self.deep_research.dual_searcher.search(search_query)
                
                # 记录搜索结果基本信息
                chunks_count = len(kbinfos.get("chunks", []))
                entities_count = len(kbinfos.get("entities", []))
                rels_count = len(kbinfos.get("relationships", []))
                self._log(f"\n[KB检索] 结果: {chunks_count}个chunks, {entities_count}个实体, {rels_count}个关系")
                
                # 为查询结果更新知识图谱
                if "chunks" in kbinfos:
                    for chunk in kbinfos.get("chunks", []):
                        chunk_id = chunk.get("chunk_id", "")
                        chunk_text = chunk.get("text", "")
                        
                        if chunk_id and chunk_text:
                            # 记录chunk作为证据
                            self.evidence_tracker.add_evidence(
                                search_step_id,
                                chunk_id,
                                chunk_text,
                                "document_chunk"
                            )
                            
                            # 尝试从chunk中提取实体和关系
                            if hasattr(self.knowledge_builder, 'extractor') and self.knowledge_builder.extractor:
                                self.knowledge_builder.extract_subgraph_from_chunk(
                                    chunk_text, 
                                    chunk_id
                                )
                
                # 检查搜索结果是否为空
                has_results = (
                    kbinfos.get("chunks", []) or 
                    kbinfos.get("entities", []) or 
                    kbinfos.get("relationships", [])
                )
                
                if not has_results:
                    self._log(f"\n[KB检索] 没有找到与'{search_query}'相关的信息")
                    no_result_msg = f"\n没有找到与'{search_query}'相关的信息。请尝试使用不同的关键词进行搜索。\n"
                    self.deep_research.thinking_engine.add_reasoning_step(no_result_msg)
                    self.deep_research.thinking_engine.add_human_message(no_result_msg)
                    think += self.deep_research.thinking_engine.remove_result_tags(no_result_msg)
                    
                    self.evidence_tracker.add_evidence(
                        search_step_id,
                        "no_results",
                        f"没有找到与查询相关的信息: {search_query}",
                        "search_result"
                    )
                    continue
                
                # 正常处理有结果的情况
                truncated_prev_reasoning = self.deep_research.thinking_engine.prepare_truncated_reasoning()
                
                # 合并块信息
                chunk_info = self.deep_research.dual_searcher._merge_results(
                    {"chunks": [], "doc_aggs": []}, 
                    kbinfos
                )
                
                # 构建提取相关信息的提示
                kb_prompt_result = "\n".join(kb_prompt(kbinfos, 4096))
                extract_prompt = RELEVANT_EXTRACTION_PROMPT.format(
                    prev_reasoning=truncated_prev_reasoning,
                    search_query=search_query,
                    document=kb_prompt_result
                )
                
                # 使用LLM提取有用信息
                self._log(f"\n[深度研究] 分析检索结果，提取有用信息")
                extraction_msg = self.llm.invoke([
                    {"role": "system", "content": extract_prompt},
                    {"role": "user", "content": f'基于当前的搜索查询"{search_query}"和前面的推理步骤，分析每个知识来源并找出有用信息。'}
                ])
                
                summary_think = extraction_msg.content if hasattr(extraction_msg, 'content') else str(extraction_msg)
                
                # 保存重要信息
                has_useful_info = (
                    "**Final Information**" in summary_think and 
                    "No helpful information found" not in summary_think
                )
                
                if has_useful_info:
                    useful_info = summary_think.split("**Final Information**")[1].strip()
                    self.deep_research.all_retrieved_info.append(useful_info)
                    
                    # 记录有用信息作为证据
                    self._log(f"\n[深度研究] 发现有用信息: {useful_info}")
                    self.evidence_tracker.add_evidence(
                        search_step_id,
                        f"useful_info_{search_query}",
                        useful_info,
                        "extracted_knowledge"
                    )
                else:
                    self._log(f"\n[深度研究] 未从检索结果中提取到有用信息")
                    self.evidence_tracker.add_evidence(
                        search_step_id,
                        f"no_useful_info_{search_query}",
                        "未从检索结果中提取到有用信息",
                        "extraction_status"
                    )
                
                # 更新推理历史
                self.deep_research.thinking_engine.add_reasoning_step(summary_think)
                self.deep_research.thinking_engine.add_human_message(summary_think)
                think += self.deep_research.thinking_engine.remove_result_tags(summary_think)
        
        # 生成最终答案
        # 确保至少执行了一次搜索
        if not self.deep_research.thinking_engine.executed_search_queries:
            self._log("\n[深度研究] 未执行任何有效搜索，无法生成答案")
            return {
                "thinking_process": think,
                "answer": f"抱歉，我无法回答关于'{query}'的问题，因为没有找到相关信息。",
                "reference": {"chunks": [], "doc_aggs": []},
                "retrieved_info": [],
                "reasoning_chain": self.evidence_tracker.get_reasoning_chain(query_id),
                "execution_logs": self.execution_logs,
            }
        
        # 使用检索到的信息生成答案
        retrieved_content = "\n\n".join(self.deep_research.all_retrieved_info)
        self._log("\n[深度研究] 生成最终答案")
        final_answer = self.deep_research._generate_final_answer(query, retrieved_content, think)
        
        # 获取知识图谱中的核心实体
        central_entities = []
        if hasattr(self.knowledge_builder, 'knowledge_graph') and self.knowledge_builder.knowledge_graph.nodes:
            central_entities = self.knowledge_builder.get_central_entities(limit=5)
        
        # 记录最终回答步骤
        final_step_id = self.evidence_tracker.add_reasoning_step(
            query_id,
            "final_answer",
            "生成最终答案"
        )
        
        self.evidence_tracker.add_evidence(
            final_step_id,
            "final_answer_evidence",
            retrieved_content,
            "synthesized_knowledge"
        )
        
        # 返回结果
        result = {
            "thinking_process": think,
            "answer": final_answer,
            "reference": chunk_info,
            "retrieved_info": self.deep_research.all_retrieved_info,
            "reasoning_chain": self.evidence_tracker.get_reasoning_chain(query_id),
            "knowledge_graph": {
                "entity_count": getattr(self.knowledge_builder.knowledge_graph, "number_of_nodes", lambda: 0)(),
                "relation_count": getattr(self.knowledge_builder.knowledge_graph, "number_of_edges", lambda: 0)(),
                "central_entities": central_entities
            },
            "execution_logs": self.execution_logs,  # 添加执行日志到返回结果
        }
        
        return result
    
    def search(self, query_input: Any) -> str:
        """
        执行增强版深度研究搜索
        
        Args:
            query_input: 搜索查询或包含查询的字典
                
        Returns:
            str: 搜索结果
        """
        overall_start = time.time()
        
        # 记录开始搜索
        self._log(f"\n[深度搜索] 开始处理查询...")
        
        # 解析输入
        if isinstance(query_input, dict) and "query" in query_input:
            query = query_input["query"]
        else:
            query = str(query_input)
        
        self._log(f"\n[深度搜索] 解析后的查询: {query}")
        
        try:
            # 执行思考过程
            self._log(f"\n[深度搜索] 开始执行思考过程")
            result = self.thinking(query)
            answer = result["answer"]
            
            # 记录总时间
            total_time = time.time() - overall_start
            self._log(f"\n[深度搜索] 完成，耗时 {total_time:.2f}秒")
            self.performance_metrics["total_time"] = total_time
            
            return answer
                
        except Exception as e:
            error_msg = f"深度研究过程中出错: {str(e)}\n{traceback.format_exc()}"
            self._log(error_msg)
            return f"抱歉，处理您的问题时遇到了错误: {str(e)}"
    
    def get_tool(self):
        """获取搜索工具"""
        from langchain_core.tools import BaseTool
        
        class DeeperResearchRetrievalTool(BaseTool):
            name : str = "deeper_research"
            description : str = "增强版深度研究工具：通过社区感知和知识图谱分析，结合多轮推理和搜索解决复杂问题。"
            
            def _run(self_tool, query: Any) -> str:
                return self.search(query)
            
            def _arun(self_tool, query: Any) -> str:
                raise NotImplementedError("异步执行未实现")
        
        return DeeperResearchRetrievalTool()
    
    async def _async_enhance_search(self, query, keywords):
        """异步执行社区感知搜索增强"""
        def enhance_wrapper():
            return self.community_search.enhance_search(query, keywords)
        
        return await asyncio.get_event_loop().run_in_executor(None, enhance_wrapper)

    async def _async_build_graph(self, query, entities):
        """异步构建知识图谱"""
        def build_wrapper():
            return self.knowledge_builder.build_query_graph(query, entities, depth=1)
        
        return await asyncio.get_event_loop().run_in_executor(None, build_wrapper)

    
    async def thinking_stream(self, query: str) -> AsyncGenerator[str, None]:
        """
        执行带流式输出的增强深度研究
        
        Args:
            query: 用户问题
                    
        Yields:
            思考步骤和最终答案
        """
        # 清空执行日志
        self.execution_logs = []
        self._log(f"\n[深度研究] 开始处理查询: {query}")
        
        # 向用户发送初始状态消息
        yield "\n**正在分析您的问题**...\n"
        
        # 提取关键词
        keywords = self.deep_research.extract_keywords(query)
        self._log(f"\n[深度研究] 提取关键词: {keywords}")
        
        # 开始新的查询跟踪
        query_id = self.evidence_tracker.start_new_query(query, keywords)
        
        # 步骤1: 社区感知增强
        yield "\n**正在分析相关知识社区**...\n"
        community_context = await self._async_enhance_search(query, keywords)
        
        # 步骤2: 使用社区信息增强搜索策略
        search_strategy = community_context.get("search_strategy", {})
        follow_up_queries = search_strategy.get("follow_up_queries", [])
        
        if follow_up_queries:
            query_msg = f"\n**发现潜在的深入探索方向**: {', '.join(follow_up_queries[:2])}\n"
            self._log(query_msg)
            yield query_msg
        
        # 添加增强搜索策略的推理步骤
        strategy_step_id = self.evidence_tracker.add_reasoning_step(
            query_id, 
            "knowledge_community_analysis", 
            f"基于社区分析，识别了关键实体和相关查询策略。将探索以下后续查询: {follow_up_queries}"
        )
        
        # 记录社区信息作为证据
        community_summaries = community_context.get("community_info", {}).get("summaries", [])
        if community_summaries:
            comm_msg = f"\n**找到 {len(community_summaries)} 个相关知识社区**\n"
            self._log(comm_msg)
            yield comm_msg
            
            for i, summary in enumerate(community_summaries[:2]):
                short_summary = summary[:100] + "..." if len(summary) > 100 else summary
                self._log(f"\n[深度研究] 社区 {i+1} 摘要: {short_summary}")
                
                self.evidence_tracker.add_evidence(
                    strategy_step_id,
                    f"community_summary_{i}",
                    summary,
                    "community_knowledge"
                )
        
        # 步骤3: 构建初始知识图谱
        initial_entities = search_strategy.get("focus_entities", [])
        if initial_entities:
            kg_msg = f"\n**正在构建相关知识图谱**...\n"
            self._log(kg_msg)
            yield kg_msg
            
            # 异步构建知识图谱
            await self._async_build_graph(query, initial_entities)
            
            # 获取核心实体
            central_entities = self.knowledge_builder.get_central_entities(limit=5)
            central_entity_ids = [e["id"] for e in central_entities]
            
            if central_entity_ids:
                central_msg = f"\n**识别出核心相关实体**: {', '.join(central_entity_ids[:3])}\n"
                yield central_msg
            
            # 添加知识图谱分析步骤
            kg_step_id = self.evidence_tracker.add_reasoning_step(
                query_id,
                "knowledge_graph_analysis",
                f"构建了初始知识图谱，识别出核心实体: {central_entity_ids}"
            )
            
            # 记录知识图谱信息作为证据
            kg_info = {
                "entity_count": self.knowledge_builder.knowledge_graph.number_of_nodes(),
                "relation_count": self.knowledge_builder.knowledge_graph.number_of_edges(),
                "central_entities": central_entity_ids
            }
            
            self.evidence_tracker.add_evidence(
                kg_step_id,
                "knowledge_graph",
                json.dumps(kg_info),
                "graph_structure"
            )
        
        # 步骤4: 使用深度研究工具的流式思考过程
        yield "\n**正在进行深度研究**...\n"
        
        # 调用deep_research工具的流式API
        async for chunk in self.deep_research.thinking_stream(query):
            if isinstance(chunk, dict) and "answer" in chunk:
                # 这是最终答案
                final_answer = chunk["answer"]
                thinking_process = chunk["thinking"]
                break
            else:
                # 其他思考过程
                yield chunk
        
        # 获取知识图谱中的核心实体
        central_entities = []
        if hasattr(self.knowledge_builder, 'knowledge_graph') and self.knowledge_builder.knowledge_graph.nodes:
            central_entities = self.knowledge_builder.get_central_entities(limit=5)
        
        # 记录最终回答步骤
        final_step_id = self.evidence_tracker.add_reasoning_step(
            query_id,
            "final_answer",
            "生成最终答案"
        )
        
        # 添加最终答案作为证据
        clean_answer = re.sub(r'<think>.*?</think>\s*', '', final_answer, flags=re.DOTALL)
        self.evidence_tracker.add_evidence(
            final_step_id,
            "final_answer",
            clean_answer,
            "response"
        )
        
        yield {"answer": final_answer, "thinking": thinking_process}
    
    async def search_stream(self, query_input: Any) -> AsyncGenerator[str, None]:
        """
        执行带流式输出的增强深度研究
        
        Args:
            query_input: 查询或包含查询的字典
                
        Yields:
            流式内容
        """
        overall_start = time.time()
        
        # 记录开始搜索
        self._log(f"\n[深度搜索] 开始处理查询...")
        
        # 解析输入
        if isinstance(query_input, dict) and "query" in query_input:
            query = query_input["query"]
        else:
            query = str(query_input)
        
        self._log(f"\n[深度搜索] 解析后的查询: {query}")
        
        # 检查缓存
        cache_key = f"deeper:{query}"
        cached_result = self.cache_manager.get(cache_key) if hasattr(self, "cache_manager") else None
        if cached_result:
            self._log(f"\n[深度搜索] 缓存命中，分块返回缓存结果")
            # 分块返回缓存结果 - 更自然的分块
            chunks = re.split(r'([.!?。！？]\s*)', cached_result)
            buffer = ""
            
            for i in range(0, len(chunks)):
                buffer += chunks[i]
                
                # 当缓冲区包含完整句子或达到合理大小时输出
                if (i % 2 == 1) or len(buffer) >= 80:
                    yield buffer
                    buffer = ""
                    await asyncio.sleep(0.01)
            
            # 输出任何剩余内容
            if buffer:
                yield buffer
            return
        
        try:
            # 执行思考过程流
            full_response = ""
            thinking_content = ""
            last_chunk = None
            
            # 提示用户处理开始
            yield "\n**开始深度分析您的问题**...\n"
            
            async for chunk in self.thinking_stream(query):
                if isinstance(chunk, dict) and "answer" in chunk:
                    # 这是最终结果对象
                    full_response = chunk["answer"]
                    thinking_content = chunk["thinking"]
                    last_chunk = chunk
                    
                    # 清理最终答案以移除思考过程部分
                    clean_answer = re.sub(r'<think>.*?</think>\s*', '', full_response, flags=re.DOTALL)
                    
                    # 如果有缓存管理器，缓存结果
                    if hasattr(self, "cache_manager"):
                        self.cache_manager.set(cache_key, clean_answer)
                    
                    yield clean_answer
                else:
                    # 正常返回流式块
                    yield chunk
            
            # 记录总时间
            total_time = time.time() - overall_start
            self._log(f"\n[深度搜索] 完成，耗时 {total_time:.2f}秒")
            if hasattr(self, "performance_metrics"):
                self.performance_metrics["total_time"] = total_time
                
        except Exception as e:
            error_msg = f"深度研究过程中出错: {str(e)}"
            self._log(error_msg)
            yield error_msg
    
    def get_stream_tool(self):
        """获取搜索工具"""
        from langchain_core.tools import BaseTool
        
        class DeeperResearchStreamTool(BaseTool):
            name : str = "deeper_research_stream"
            description : str = "增强版流式深度研究工具：通过社区感知和知识图谱分析，结合多轮推理和搜索解决复杂问题，支持流式输出。"
            
            def _run(self_tool, query: Any) -> AsyncGenerator:
                return self.search_stream(query)
            
            async def _arun(self_tool, query: Any) -> AsyncGenerator:
                return await self.search_stream(query)
        
        return DeeperResearchStreamTool()