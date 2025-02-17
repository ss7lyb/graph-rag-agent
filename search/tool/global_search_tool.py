import os
from typing import List
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph

from model.get_models import get_llm_model
from config.prompt import MAP_SYSTEM_PROMPT
from config.settings import gl_description

load_dotenv('../../.env')

class GlobalSearchTool:
    def __init__(self, level: int = 0, tool_description: str = gl_description):
        self.llm = get_llm_model()
        self.level = level
        self.tool_description = tool_description
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
                """),
        ])
        self.map_chain = map_prompt | self.llm | StrOutputParser()

    def _get_community_data(self) -> List[dict]:
        """Retrieve community data from Neo4j"""
        return self.graph.query(
            """
            MATCH (c:__Community__)
            WHERE c.level = $level
            RETURN {communityId:c.id, full_content:c.full_content} AS output
            """,
            params={"level": self.level},
        )

    def _search_impl(self, query: str) -> List[str]:
        """Implementation of the search functionality"""
        community_data = self._get_community_data()
        
        intermediate_results = []
        for community in tqdm(community_data, desc="Processing communities"):
            intermediate_response = self.map_chain.invoke({
                "question": query, 
                "context_data": community["output"]
            })
            intermediate_results.append(intermediate_response)
            
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