import re
import ast
from typing import List, Any
from langchain_community.graphs import Neo4jGraph
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from model.get_models import get_llm_model
from config.prompt import system_template_build_index, user_template_build_index

from dotenv import load_dotenv

load_dotenv('../.env')

class EntityMerger:
    """
    实体合并管理器，负责基于LLM决策合并相似实体。
    主要功能包括使用LLM分析实体相似性、解析合并建议，以及执行实体合并操作。
    """
    
    def __init__(self, graph: Neo4jGraph = None):
        """
        初始化实体合并管理器
        Args:
            graph: Neo4j图数据库连接，如果不提供则创建新的连接
        """
        # 初始化图数据库连接
        self.graph = graph if graph else Neo4jGraph()
        # 获取语言模型
        self.llm = get_llm_model()
        # 设置LLM处理链
        self._setup_llm_chain()

    def _setup_llm_chain(self) -> None:
        """
        设置LLM处理链，用于实体合并决策
        包括创建提示模板和构建处理链
        """
        # 检查模型能力
        if not hasattr(self.llm, 'with_structured_output'):
            print("当前LLM模型不支持结构化输出")

        # 创建提示模板
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            system_template_build_index
        )
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            user_template_build_index
        )
        
        # 构建对话链
        self.chat_prompt = ChatPromptTemplate.from_messages([
            system_message_prompt,
            MessagesPlaceholder("chat_history"),
            human_message_prompt
        ])
        
        # 创建最终的处理链
        self.chain = self.chat_prompt | self.llm

    def _convert_to_list(self, result: str) -> List[List[str]]:
        """
        将LLM返回的实体列表文本转换为Python列表
        
        Args:
            result: LLM返回的文本结果，包含实体列表
            
        Returns:
            List[List[str]]: 二维列表，每个子列表包含一组可合并的实体
        """
        # 使用正则表达式匹配所有方括号包含的内容
        list_pattern = re.compile(r'\[.*?\]')
        entity_lists = []
        
        # 解析每个匹配的列表字符串
        for match in list_pattern.findall(result):
            try:
                # 将字符串转换为Python列表
                entity_list = ast.literal_eval(match)
                # 只添加非空列表
                if entity_list:
                    entity_lists.append(entity_list)
            except Exception as e:
                print(f"解析实体列表时出错: {str(e)}, 原文本: {match}")
        
        return entity_lists

    def get_merge_suggestions(self, duplicate_candidates: List[Any]) -> List[List[str]]:
        """
        使用LLM分析并提供实体合并建议
        
        Args:
            duplicate_candidates: 潜在的重复实体候选列表
            
        Returns:
            List[List[str]]: 建议合并的实体分组列表
        """
        # 收集LLM的合并建议
        merged_entities = []
        
        # 对每组候选实体进行分析
        for candidates in duplicate_candidates:
            chat_history = []
            # 调用LLM进行分析
            answer = self.chain.invoke({
                "chat_history": chat_history,
                "entities": candidates
            })
            merged_entities.append(answer.content)
        
        # 解析并整理最终的合并建议
        results = []
        for candidates in merged_entities:
            # 将每个建议转换为列表格式
            temp = self._convert_to_list(candidates)
            results.extend(temp)
        
        return results

    def execute_merges(self, merge_groups: List[List[str]]) -> int:
        """
        执行实体合并操作
        
        Args:
            merge_groups: 要合并的实体分组列表
            
        Returns:
            int: 合并操作影响的节点数量
        """
        # 执行Neo4j合并操作
        result = self.graph.query("""
        UNWIND $data AS candidates
        CALL {
          WITH candidates
          MATCH (e:__Entity__) WHERE e.id IN candidates
          RETURN collect(e) AS nodes
        }
        CALL apoc.refactor.mergeNodes(nodes, {properties: {
            `.*`: 'discard'
        }})
        YIELD node
        RETURN count(*) as merged_count
        """, params={"data": merge_groups})
        
        return result[0]["merged_count"] if result else 0

    def process_duplicates(self, duplicate_candidates: List[Any]) -> int:
        """
        处理重复实体的完整流程，包括获取合并建议和执行合并
        
        Args:
            duplicate_candidates: 潜在的重复实体候选列表
            
        Returns:
            int: 合并的实体数量
        """
        # 获取合并建议
        merge_groups = self.get_merge_suggestions(duplicate_candidates)
        
        # 如果有建议的合并组，执行合并
        if merge_groups:
            return self.execute_merges(merge_groups)
        return 0