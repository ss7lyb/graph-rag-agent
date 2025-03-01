import os
import hashlib
from typing import List
from dotenv import load_dotenv
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph

from model.get_models import get_llm_model
from config.prompt import MAP_SYSTEM_PROMPT
from config.settings import gl_description

load_dotenv()

class GlobalSearchCache:
    """全局搜索缓存"""
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
    
    def get_key(self, query: str, keywords: List[str] = None):
        """生成缓存键"""
        key_str = query
        if keywords:
            key_str += "||" + ",".join(sorted(keywords))
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, query: str, keywords: List[str] = None):
        """获取缓存结果"""
        key = self.get_key(query, keywords)
        return self.cache.get(key)
    
    def set(self, query: str, keywords: List[str], result: List[str]):
        """设置缓存结果"""
        key = self.get_key(query, keywords)
        
        # 如果缓存已满，移除最早的项
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = result

class GlobalSearchTool:
    def __init__(self, level: int = 0, tool_description: str = gl_description):
        self.llm = get_llm_model()
        self.level = level
        self.tool_description = tool_description
        self.cache = GlobalSearchCache()
        self._setup_neo4j()
        self._setup_chains()

    def _setup_neo4j(self):
        self.graph = Neo4jGraph(
            url=os.getenv('NEO4J_URI'),
            username=os.getenv('NEO4J_USERNAME'),
            password=os.getenv('NEO4J_PASSWORD'),
            refresh_schema=False,
        )

    def _setup_chains(self):
        """Setup LLM chains for mapping"""
        map_prompt = ChatPromptTemplate.from_messages([
            ("system", MAP_SYSTEM_PROMPT),
            ("human", """
                ---数据表格--- 
                {context_data}
                
                用户的问题是：
                {question}
                """
            ),
        ])
        self.map_chain = map_prompt | self.llm | StrOutputParser()

    def _get_community_data(self, keywords: List[str] = None) -> List[dict]:
        """使用关键词检索社区数据"""
        cypher_query = """
        MATCH (c:__Community__)
        WHERE c.level = $level
        """
        
        params = {"level": self.level}
        
        # 如果提供了关键词，使用它们过滤社区
        if keywords and len(keywords) > 0:
            keywords_condition = []
            for i, keyword in enumerate(keywords):
                keyword_param = f"keyword{i}"
                keywords_condition.append(f"c.full_content CONTAINS ${keyword_param}")
                params[keyword_param] = keyword
            
            if keywords_condition:
                cypher_query += " AND (" + " OR ".join(keywords_condition) + ")"
        
        # 添加排序和返回语句
        cypher_query += """
        WITH c
        ORDER BY c.community_rank DESC, c.weight DESC
        LIMIT 20
        RETURN {communityId: c.id, full_content: c.full_content} AS output
        """
        
        # 执行查询
        return self.graph.query(cypher_query, params=params)

    def _process_community_batch(self, query: str, batch: List[dict]) -> str:
        """处理社区批次"""
        # 合并批次内的社区数据
        combined_data = []
        for item in batch:
            combined_data.append(f"社区ID: {item['output']['communityId']}\n内容: {item['output']['full_content']}")
        
        batch_context = "\n---\n".join(combined_data)
        
        # 一次性处理整个批次
        return self.map_chain.invoke({
            "question": query, 
            "context_data": batch_context
        })

    def _search_impl(self, query: str) -> List[str]:
        """Implementation of the search functionality with batch processing"""
        # 提取消息中的关键词，如果有的话
        keywords = []
        if isinstance(query, dict) and "query" in query:
            keywords = query.get("keywords", [])
            query = query["query"]
        
        # Check cache
        cached_result = self.cache.get(query, keywords)
        if cached_result:
            return cached_result
        
        # Get community data
        community_data = self._get_community_data(keywords)
        
        # If no data found, return empty results
        if not community_data:
            return []
        
        # Define batch size and process communities in batches
        batch_size = 5  # Process 5 communities at a time
        
        intermediate_results = []
        
        # 串行处理批次以避免并发问题
        for i in range(0, len(community_data), batch_size):
            batch = community_data[i:i+batch_size]
            try:
                batch_result = self._process_community_batch(query, batch)
                if batch_result and len(batch_result.strip()) > 0:
                    intermediate_results.append(batch_result)
            except Exception as e:
                print(f"批处理失败: {e}")
        
        # Cache results
        self.cache.set(query, keywords if keywords else [], intermediate_results)
        
        return intermediate_results

    @property
    def search(self):
        class DynamicSearchTool(BaseTool):
            name = "global_retriever"
            description = self.tool_description
            
            def _run(self_tool, query: str) -> List[str]:
                return self._search_impl(query)
            
            def _arun(self_tool, query: str) -> List[str]:
                raise NotImplementedError("Async not implemented")
            
        return DynamicSearchTool()