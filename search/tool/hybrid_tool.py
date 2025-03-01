import os
import hashlib
from typing import List, Dict, Any, Optional
import time
import pandas as pd
from neo4j import GraphDatabase, Result

from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph

from model.get_models import get_llm_model, get_embeddings_model
from config.prompt import LC_SYSTEM_PROMPT
from config.settings import lc_description, gl_description, response_type

class HybridSearchCache:
    """改进的搜索缓存，支持双级关键词"""
    def __init__(self, max_size: int = 200, cache_dir: str = "./cache/search"):
        self.cache = {}
        self.max_size = max_size
        self.cache_dir = cache_dir
        
        # 确保缓存目录存在
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # 加载磁盘缓存
        self._load_from_disk()
    
    def _get_cache_path(self, key: str) -> str:
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f"{key}.txt")
    
    def _load_from_disk(self):
        """从磁盘加载缓存"""
        if os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                if filename.endswith(".txt"):
                    key = filename[:-4]  # 移除.txt后缀
                    try:
                        with open(os.path.join(self.cache_dir, filename), 'r', encoding='utf-8') as f:
                            self.cache[key] = f.read()
                    except Exception as e:
                        print(f"加载缓存文件 {filename} 时出错: {e}")
    
    def get_key(self, query: str, low_level_keywords: List[str] = None, high_level_keywords: List[str] = None) -> str:
        """生成缓存键，考虑双级关键词"""
        key_parts = [query]
        
        if low_level_keywords:
            key_parts.append("low:" + ",".join(sorted(low_level_keywords)))
            
        if high_level_keywords:
            key_parts.append("high:" + ",".join(sorted(high_level_keywords)))
            
        key_str = "||".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, query: str, low_level_keywords: List[str] = None, high_level_keywords: List[str] = None) -> Optional[str]:
        """获取缓存结果"""
        key = self.get_key(query, low_level_keywords, high_level_keywords)
        return self.cache.get(key)
    
    def set(self, query: str, result: str, low_level_keywords: List[str] = None, high_level_keywords: List[str] = None) -> None:
        """设置缓存结果"""
        key = self.get_key(query, low_level_keywords, high_level_keywords)
        
        # 如果缓存已满，移除最早的项
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            # 删除磁盘上的缓存文件
            cache_path = self._get_cache_path(oldest_key)
            if os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                except Exception as e:
                    print(f"删除缓存文件 {cache_path} 时出错: {e}")
        
        # 更新内存缓存
        self.cache[key] = result
        
        # 写入磁盘缓存
        try:
            with open(self._get_cache_path(key), 'w', encoding='utf-8') as f:
                f.write(result)
        except Exception as e:
            print(f"写入缓存文件时出错: {e}")

class HybridSearchTool:
    """
    增强型搜索工具，实现类似LightRAG的双级检索策略
    结合了局部细节检索和全局主题检索
    """
    
    def __init__(self):
        """初始化增强型搜索工具"""
        # 初始化模型
        self.llm = get_llm_model()
        self.embeddings = get_embeddings_model()
        
        # 缓存设置
        self.cache = HybridSearchCache()
        
        # Neo4j连接设置
        self._setup_neo4j()
        
        # 初始化查询链
        self._setup_chains()
        
        # 检索参数
        self.entity_limit = 15        # 最大检索实体数量
        self.max_hop_distance = 2     # 最大跳数（关系扩展）
        self.top_communities = 3      # 检索社区数量
        self.batch_size = 10          # 批处理大小
        self.community_level = 0      # 默认社区等级
        
        # 性能监控
        self.query_time = 0
        self.llm_time = 0
    
    def _setup_neo4j(self):
        """设置Neo4j连接"""
        # 获取环境变量
        neo4j_uri = os.getenv('NEO4J_URI')
        neo4j_username = os.getenv('NEO4J_USERNAME')
        neo4j_password = os.getenv('NEO4J_PASSWORD')
        
        # 初始化Neo4j图连接
        self.graph = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
            refresh_schema=False,
        )
        
        # 初始化Neo4j驱动，用于直接执行查询
        self.driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_username, neo4j_password)
        )
        
        # 不使用Neo4jVector，因为配置不兼容
        # 我们将使用自定义实现的向量搜索
        self.vector_store = None
    
    def _setup_chains(self):
        """设置处理链"""
        # 创建主查询处理链 - 用于生成最终答案
        self.query_prompt = ChatPromptTemplate.from_messages([
            ("system", LC_SYSTEM_PROMPT),
            ("human", """
                ---分析报告--- 
                请注意，以下内容组合了低级详细信息和高级主题概念。

                ## 低级内容（实体详细信息）:
                {low_level}
                
                ## 高级内容（主题和概念）:
                {high_level}

                用户的问题是：
                {query}
                
                请综合利用上述信息回答问题，确保回答全面且有深度。
                回答格式应包含：
                1. 主要内容（使用清晰的段落展示）
                2. 在末尾标明引用的数据来源
                """
            )
        ])
        
        # 链接到LLM
        self.query_chain = self.query_prompt | self.llm | StrOutputParser()
        
        # 关键词提取链
        self.keyword_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专门从用户查询中提取搜索关键词的助手。你需要将关键词分为两类：
                1. 低级关键词：具体实体名称、人物、地点、具体事件等
                2. 高级关键词：主题、概念、关系类型等
                
                返回格式必须是JSON格式：
                {{
                    "low_level": ["关键词1", "关键词2", ...], 
                    "high_level": ["关键词1", "关键词2", ...]
                }}
                
                注意：
                - 每类提取3-5个关键词即可
                - 不要添加任何解释或其他文本，只返回JSON
                - 如果某类无关键词，则返回空列表
                """),
            ("human", "{query}")
        ])
        
        self.keyword_chain = self.keyword_prompt | self.llm | StrOutputParser()
    
    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """从查询中提取双级关键词"""
        try:
            llm_start = time.time()
            
            # 尝试解析JSON结果
            result = self.keyword_chain.invoke({"query": query})
            import json
            keywords = json.loads(result)
            
            self.llm_time += time.time() - llm_start
            
            # 确保包含必要的键
            if not isinstance(keywords, dict):
                keywords = {}
            if "low_level" not in keywords:
                keywords["low_level"] = []
            if "high_level" not in keywords:
                keywords["high_level"] = []
                
            return keywords
            
        except Exception as e:
            print(f"关键词提取失败: {e}")
            # 返回空字典作为默认值
            return {"low_level": [], "high_level": []}
    
    def db_query(self, cypher: str, params: Dict[str, Any] = {}) -> pd.DataFrame:
        """执行Cypher查询并返回结果"""
        return self.driver.execute_query(
            cypher,
            parameters_=params,
            result_transformer_=Result.to_df
        )
    
    def _vector_search(self, query: str, limit: int = 5) -> List[str]:
        """
        自定义向量搜索方法
        
        参数:
            query: 搜索查询
            limit: 最大返回结果数
            
        返回:
            List[str]: 匹配实体ID列表
        """
        try:
            # 生成查询的嵌入向量
            query_embedding = self.embeddings.embed_query(query)
            
            # 构建Neo4j向量搜索查询
            cypher = """
            CALL db.index.vector.queryNodes('vector', $limit, $embedding)
            YIELD node, score
            RETURN node.id AS id, score
            ORDER BY score DESC
            """
            
            # 执行搜索
            results = self.db_query(cypher, {
                "embedding": query_embedding,
                "limit": limit
            })
            
            # 提取实体ID
            if not results.empty:
                return results['id'].tolist()
            else:
                return []
                
        except Exception as e:
            print(f"向量搜索失败: {e}")
            # 如果向量搜索失败，尝试使用文本搜索作为备用
            return self._fallback_text_search(query, limit)

    def _fallback_text_search(self, query: str, limit: int = 5) -> List[str]:
        """
        基于文本匹配的备用搜索方法
        
        参数:
            query: 搜索查询
            limit: 最大返回结果数
            
        返回:
            List[str]: 匹配实体ID列表
        """
        try:
            # 构建全文搜索查询
            cypher = """
            MATCH (e:__Entity__)
            WHERE e.id CONTAINS $query OR e.description CONTAINS $query
            RETURN e.id AS id
            LIMIT $limit
            """
            
            results = self.db_query(cypher, {
                "query": query,
                "limit": limit
            })
            
            if not results.empty:
                return results['id'].tolist()
            else:
                return []
                
        except Exception as e:
            print(f"文本搜索也失败: {e}")
            return []
    
    def _retrieve_low_level_content(self, query: str, keywords: List[str]) -> str:
        """检索低级内容（具体实体和关系）"""
        query_start = time.time()
        
        # 首先使用关键词查询获取相关实体
        entity_ids = []
        
        if keywords:
            keyword_params = {}
            keyword_conditions = []
            
            for i, keyword in enumerate(keywords):
                param_name = f"keyword{i}"
                keyword_params[param_name] = keyword
                keyword_conditions.append(f"e.id CONTAINS ${param_name} OR e.description CONTAINS ${param_name}")
            
            # 构建查询
            if keyword_conditions:
                keyword_query = """
                MATCH (e:__Entity__)
                WHERE """ + " OR ".join(keyword_conditions) + """
                RETURN e.id AS id
                LIMIT $limit
                """
                
                try:
                    keyword_results = self.db_query(keyword_query, 
                                                {**keyword_params, "limit": self.entity_limit})
                    if not keyword_results.empty:
                        entity_ids = keyword_results['id'].tolist()
                except Exception as e:
                    print(f"关键词查询失败: {e}")
        
        # 如果关键词搜索没有结果或没有提供关键词，尝试使用向量搜索
        if not entity_ids:
            try:
                # 使用我们的自定义向量搜索方法
                vector_entity_ids = self._vector_search(query, limit=self.entity_limit)
                if vector_entity_ids:
                    entity_ids = vector_entity_ids
            except Exception as e:
                print(f"向量搜索失败: {e}")
        
        # 如果仍然没有实体，使用基本文本匹配
        if not entity_ids:
            try:
                entity_ids = self._fallback_text_search(query, limit=self.entity_limit)
            except Exception as e:
                print(f"文本搜索失败: {e}")
        
        # 如果仍然没有实体，返回空内容
        if not entity_ids:
            self.query_time += time.time() - query_start
            return "没有找到相关的低级内容。"
        
        # 获取实体信息 - 不使用多跳关系以避免复杂查询
        entity_query = """
        // 从种子实体开始
        MATCH (e:__Entity__)
        WHERE e.id IN $entity_ids
        
        RETURN collect({
            id: e.id, 
            type: CASE WHEN size(labels(e)) > 1 
                     THEN [lbl IN labels(e) WHERE lbl <> '__Entity__'][0] 
                     ELSE 'Unknown' 
                  END, 
            description: e.description
        }) AS entities
        """
        
        # 获取关系信息 - 分别查询，避免复杂路径
        relation_query = """
        // 查找实体间的关系
        MATCH (e1:__Entity__)-[r]-(e2:__Entity__)
        WHERE e1.id IN $entity_ids 
          AND e2.id IN $entity_ids
          AND e1.id < e2.id  // 避免重复关系
        
        RETURN collect({
            start: e1.id, 
            type: type(r), 
            end: e2.id,
            description: CASE WHEN r.description IS NULL THEN '' ELSE r.description END
        }) AS relationships
        """
        
        # 获取文本块信息
        chunk_query = """
        // 查找包含这些实体的文本块
        MATCH (c:__Chunk__)-[:MENTIONS]->(e:__Entity__)
        WHERE e.id IN $entity_ids
        
        RETURN collect(DISTINCT {
            id: c.id, 
            text: c.text
        })[0..5] AS chunks
        """
        
        try:
            # 获取实体信息
            entity_results = self.db_query(entity_query, {"entity_ids": entity_ids})
            
            # 获取关系信息
            relation_results = self.db_query(relation_query, {"entity_ids": entity_ids})
            
            # 获取文本块信息
            chunk_results = self.db_query(chunk_query, {"entity_ids": entity_ids})
            
            self.query_time += time.time() - query_start
            
            # 构建结果
            low_level = []
            
            # 添加实体信息
            if not entity_results.empty and 'entities' in entity_results.columns:
                entities = entity_results.iloc[0]['entities']
                if entities:
                    low_level.append("### 相关实体")
                    for entity in entities:
                        entity_desc = f"- **{entity['id']}** ({entity['type']}): {entity['description']}"
                        low_level.append(entity_desc)
            
            # 添加关系信息
            if not relation_results.empty and 'relationships' in relation_results.columns:
                relationships = relation_results.iloc[0]['relationships']
                if relationships:
                    low_level.append("\n### 实体关系")
                    for rel in relationships:
                        rel_desc = f"- **{rel['start']}** -{rel['type']}-> **{rel['end']}**: {rel['description']}"
                        low_level.append(rel_desc)
            
            # 添加文本块信息
            if not chunk_results.empty and 'chunks' in chunk_results.columns:
                chunks = chunk_results.iloc[0]['chunks']
                if chunks:
                    low_level.append("\n### 相关文本")
                    for chunk in chunks:
                        chunk_text = f"- ID: {chunk['id']}\n  内容: {chunk['text']}"
                        low_level.append(chunk_text)
            
            if not low_level:
                return "没有找到相关的低级内容。"
                
            return "\n".join(low_level)
        except Exception as e:
            self.query_time += time.time() - query_start
            print(f"实体查询失败: {e}")
            return "查询实体信息时出错。"
    
    def _retrieve_high_level_content(self, query: str, keywords: List[str]) -> str:
        """检索高级内容（社区和主题概念）"""
        query_start = time.time()
        
        # 构建关键词条件
        keyword_conditions = []
        params = {"level": self.community_level, "limit": self.top_communities}
        
        if keywords:
            for i, keyword in enumerate(keywords):
                param_name = f"keyword{i}"
                params[param_name] = keyword
                keyword_conditions.append(f"c.summary CONTAINS ${param_name} OR c.full_content CONTAINS ${param_name}")
        
        # 构建查询
        community_query = """
        // 使用关键词过滤社区
        MATCH (c:__Community__ {level: $level})
        """
        
        if keyword_conditions:
            community_query += "WHERE " + " OR ".join(keyword_conditions)
        else:
            # 如果没有关键词，则使用查询文本
            params["query"] = query
            community_query += "WHERE c.summary CONTAINS $query OR c.full_content CONTAINS $query"
        
        # 添加排序和限制
        community_query += """
        WITH c
        ORDER BY CASE WHEN c.community_rank IS NULL THEN 0 ELSE c.community_rank END DESC
        LIMIT $limit
        RETURN c.id AS id, c.summary AS summary
        """
        
        try:
            community_results = self.db_query(community_query, params)
            
            self.query_time += time.time() - query_start
            
            # 处理结果
            if community_results.empty:
                return "没有找到相关的高级内容。"
                
            # 构建格式化的高级内容
            high_level = ["### 相关主题概念"]
            
            for _, row in community_results.iterrows():
                community_desc = f"- **社区 {row['id']}**:\n  {row['summary']}"
                high_level.append(community_desc)
            
            return "\n".join(high_level)
        except Exception as e:
            self.query_time += time.time() - query_start
            print(f"社区查询失败: {e}")
            return "查询社区信息时出错。"
    
    def search(self, query_input: Any) -> str:
        """
        执行增强型搜索，结合低级和高级内容
        
        参数:
            query_input: 字符串查询或包含查询和关键词的字典
            
        返回:
            str: 生成的最终答案
        """
        # 解析输入
        if isinstance(query_input, dict) and "query" in query_input:
            query = query_input["query"]
            # 支持直接传入分类的关键词
            low_keywords = query_input.get("low_level_keywords", [])
            high_keywords = query_input.get("high_level_keywords", [])
        else:
            query = str(query_input)
            # 提取关键词
            keywords = self.extract_keywords(query)
            low_keywords = keywords.get("low_level", [])
            high_keywords = keywords.get("high_level", [])
        
        # 检查缓存
        cached_result = self.cache.get(query, low_keywords, high_keywords)
        if cached_result:
            return cached_result
        
        try:
            # 1. 检索低级内容（实体和关系）
            low_level_content = self._retrieve_low_level_content(query, low_keywords)
            
            # 2. 检索高级内容（社区和主题）
            high_level_content = self._retrieve_high_level_content(query, high_keywords)
            
            # 3. 生成最终答案
            llm_start = time.time()
            
            # 添加需要的response_type参数
            result = self.query_chain.invoke({
                "query": query,
                "low_level": low_level_content,
                "high_level": high_level_content,
                "response_type": response_type
            })
            
            self.llm_time += time.time() - llm_start
            
            # 缓存结果
            self.cache.set(query, result, low_keywords, high_keywords)
            
            return result
            
        except Exception as e:
            error_msg = f"搜索过程中出现错误: {str(e)}"
            print(error_msg)
            return error_msg
    
    def get_tool(self) -> BaseTool:
        """获取本地检索工具"""
        class DualLevelSearchTool(BaseTool):
            name = "enhanced_search_tool"
            description = lc_description
            
            def _run(self_tool, query: Any) -> str:
                return self.search(query)
            
            def _arun(self_tool, query: Any) -> str:
                raise NotImplementedError("异步执行未实现")
                
        return DualLevelSearchTool()
    
    def get_global_tool(self) -> BaseTool:
        """获取全局搜索工具"""
        class GlobalSearchTool(BaseTool):
            name = "global_retriever"
            description = gl_description
            
            def _run(self_tool, query: Any) -> str:
                # 设置为仅使用高级内容
                if isinstance(query, dict) and "query" in query:
                    original_query = query["query"]
                    keywords = query.get("keywords", [])
                    # 转换为高级关键词
                    high_keywords = keywords
                    query = {
                        "query": original_query,
                        "high_level_keywords": high_keywords,
                        "low_level_keywords": []  # 不使用低级关键词
                    }
                else:
                    # 提取关键词
                    keywords = self.extract_keywords(str(query))
                    query = {
                        "query": str(query),
                        "high_level_keywords": keywords.get("high_level", []),
                        "low_level_keywords": []
                    }
                
                return self.search(query)
            
            def _arun(self_tool, query: Any) -> str:
                raise NotImplementedError("异步执行未实现")
                
        return GlobalSearchTool()
    
    def close(self):
        """关闭资源连接"""
        if hasattr(self, 'driver'):
            self.driver.close()
