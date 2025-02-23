from abc import ABC, abstractmethod
from typing import List, Dict
from langchain_community.graphs import Neo4jGraph
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from model.get_models import get_llm_model
from config.prompt import community_template

class BaseCommunityDescriber:
    """基础的社区信息格式化工具"""
    
    @staticmethod
    def prepare_string(data: Dict) -> str:
        """将社区信息转换为可读的字符串格式"""
        nodes_str = "Nodes are:\n"
        for node in data['nodes']:
            node_id = node['id']
            node_type = node['type']
            node_description = (
                f", description: {node['description']}"
                if 'description' in node and node['description']
                else ""
            )
            nodes_str += f"id: {node_id}, type: {node_type}{node_description}\n"

        rels_str = "Relationships are:\n"
        for rel in data['rels']:
            start = rel['start']
            end = rel['end']
            rel_type = rel['type']
            description = (
                f", description: {rel['description']}"
                if 'description' in rel and rel['description']
                else ""
            )
            rels_str += f"({start})-[:{rel_type}]->({end}){description}\n"

        return nodes_str + "\n" + rels_str

class BaseCommunityRanker:
    """基础的社区权重计算工具"""
    
    def __init__(self, graph: Neo4jGraph):
        self.graph = graph
    
    def calculate_ranks(self) -> None:
        """计算社区的权重"""
        self.graph.query("""
        MATCH (c:`__Community__`)<-[:IN_COMMUNITY*]-(:`__Entity__`)<-[:MENTIONS]-(d:`__Chunk__`)
        WITH c, count(distinct d) AS rank
        SET c.community_rank = rank;
        """)

class BaseCommunityCollector(ABC):
    """基础的社区信息收集工具"""
    
    def __init__(self, graph: Neo4jGraph):
        self.graph = graph
    
    @abstractmethod
    def collect_info(self) -> List[Dict]:
        """收集社区信息的抽象方法"""
        pass

class BaseCommunityStorer:
    """基础的社区信息存储工具"""
    
    def __init__(self, graph: Neo4jGraph):
        self.graph = graph
    
    def store_summaries(self, summaries: List[Dict]) -> None:
        """存储社区摘要"""
        self.graph.query("""
        UNWIND $data AS row
        MERGE (c:__Community__ {id:row.community})
        SET c.summary = row.summary, c.full_content = row.full_content
        """, params={"data": summaries})

class BaseSummarizer(ABC):
    """
    社区摘要生成器的基类，定义了基本的摘要生成流程。
    """
    
    def __init__(self, graph: Neo4jGraph):
        self.graph = graph
        self.llm = get_llm_model()
        self.describer = BaseCommunityDescriber()
        self.ranker = BaseCommunityRanker(graph)
        self.storer = BaseCommunityStorer(graph)
        self._setup_llm_chain()

    def _setup_llm_chain(self) -> None:
        """设置LLM处理链"""
        community_prompt = ChatPromptTemplate.from_messages([
            ("system", "给定一个输入三元组，生成信息摘要。没有序言。"),
            ("human", community_template),
        ])
        self.community_chain = community_prompt | self.llm | StrOutputParser()

    def _process_community(self, community: Dict) -> Dict:
        """处理单个社区并生成摘要"""
        stringify_info = self.describer.prepare_string(community)
        summary = self.community_chain.invoke({'community_info': stringify_info})
        return {
            "community": community['communityId'],
            "summary": summary,
            "full_content": stringify_info
        }

    @abstractmethod
    def collect_community_info(self) -> List[Dict]:
        """收集社区信息的抽象方法"""
        pass

    def process_communities(self) -> List[Dict]:
        """处理所有社区的完整流程"""
        try:
            # 计算社区权重
            self.ranker.calculate_ranks()
            
            # 收集社区信息
            community_info = self.collect_community_info()
            
            # 生成摘要
            summaries = []
            for info in community_info:
                result = self._process_community(info)
                summaries.append(result)
            
            # 保存摘要
            self.storer.store_summaries(summaries)
            
            return summaries
            
        except Exception as e:
            print(f"处理社区摘要时出错: {str(e)}")
            raise

class LeidenSummarizer(BaseSummarizer):
    """
    Leiden算法的社区摘要生成器。
    考虑Leiden算法生成的多层级社区结构。
    """
    
    def collect_community_info(self) -> List[Dict]:
        """收集Leiden算法生成的社区信息"""
        return self.graph.query("""
        // 找到最底层(level=0)的社区
        MATCH (c:`__Community__` {level: 0})
        MATCH (c)<-[:IN_COMMUNITY]-(e:__Entity__)
        WITH c, collect(e) as nodes
        WHERE size(nodes) > 1
        // 获取实体间的关系
        MATCH (n1:__Entity__)
        WHERE n1 IN nodes
        MATCH (n2:__Entity__)
        WHERE n2 IN nodes AND id(n1) < id(n2)
        MATCH (n1)-[r]->(n2)
        WITH c, nodes, collect(distinct r) as rels
        RETURN c.id AS communityId,
               [n in nodes | {
                   id: n.id, 
                   description: n.description, 
                   type: [el in labels(n) WHERE el <> '__Entity__'][0]
               }] AS nodes,
               [r in rels | {
                   start: startNode(r).id, 
                   type: type(r), 
                   end: endNode(r).id, 
                   description: r.description
               }] AS rels
        """)

class SLLPASummarizer(BaseSummarizer):
    """
    SLLPA算法的社区摘要生成器。
    SLLPA算法生成的社区没有层级结构，所有社区都在同一层级(level=0)。
    """
    
    def collect_community_info(self) -> List[Dict]:
        """收集SLLPA算法生成的社区信息"""
        # SLLPA算法生成的社区都在同一层级
        return self.graph.query("""
        MATCH (c:`__Community__`)<-[:IN_COMMUNITY*]-(e:__Entity__)
        WHERE c.level = 0  // SLLPA的所有社区都在level 0
        WITH c, collect(e) AS nodes
        WHERE size(nodes) > 1  // 只选择包含多个节点的社区
        CALL apoc.path.subgraphAll(nodes[0], {
            whitelistNodes:nodes
        })
        YIELD relationships
        RETURN c.id AS communityId,
               [n in nodes | {
                   id: n.id, 
                   description: n.description, 
                   type: [el in labels(n) WHERE el <> '__Entity__'][0]
               }] AS nodes,
               [r in relationships | {
                   start: startNode(r).id, 
                   type: type(r), 
                   end: endNode(r).id, 
                   description: r.description
               }] AS rels
        """)

class CommunitySummarizerFactory:
    """社区摘要生成器工厂类"""
    
    @staticmethod
    def create_summarizer(algorithm: str, graph: Neo4jGraph) -> BaseSummarizer:
        """
        根据指定的算法类型创建相应的摘要生成器
        
        Args:
            algorithm: 算法类型 ('leiden' 或 'sllpa')
            graph: Neo4j图数据库连接
            
        Returns:
            BaseSummarizer: 具体的摘要生成器实例
        """
        if algorithm.lower() == 'leiden':
            return LeidenSummarizer(graph)
        elif algorithm.lower() == 'sllpa':
            return SLLPASummarizer(graph)
        else:
            raise ValueError(f"不支持的算法类型: {algorithm}")