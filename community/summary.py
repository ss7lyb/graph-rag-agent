from abc import ABC, abstractmethod
from typing import List, Dict
from langchain_community.graphs import Neo4jGraph
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from model.get_models import get_llm_model
from config.prompt import community_template
import concurrent.futures
import time
import os

class BaseCommunityDescriber:
    """基础的社区信息格式化工具"""
    
    @staticmethod
    def prepare_string(data: Dict) -> str:
        """将社区信息转换为可读的字符串格式"""
        try:
            nodes_str = "Nodes are:\n"
            for node in data.get('nodes', []):
                node_id = node.get('id', 'unknown_id')
                node_type = node.get('type', 'unknown_type')
                node_description = (
                    f", description: {node['description']}"
                    if 'description' in node and node['description']
                    else ""
                )
                nodes_str += f"id: {node_id}, type: {node_type}{node_description}\n"

            rels_str = "Relationships are:\n"
            for rel in data.get('rels', []):
                start = rel.get('start', 'unknown_start')
                end = rel.get('end', 'unknown_end')
                rel_type = rel.get('type', 'unknown_type')
                description = (
                    f", description: {rel['description']}"
                    if 'description' in rel and rel['description']
                    else ""
                )
                rels_str += f"({start})-[:{rel_type}]->({end}){description}\n"

            return nodes_str + "\n" + rels_str
        except Exception as e:
            print(f"格式化社区信息时出错: {e}")
            # 提供备用格式以确保流程可以继续
            return f"Error formatting community data: {str(e)}\nOriginal data: {str(data)}"

class BaseCommunityRanker:
    """基础的社区权重计算工具"""
    
    def __init__(self, graph: Neo4jGraph):
        self.graph = graph
    
    def calculate_ranks(self) -> None:
        """计算社区的权重"""
        start_time = time.time()
        print("计算社区权重...")
        
        try:
            result = self.graph.query("""
            MATCH (c:`__Community__`)<-[:IN_COMMUNITY*]-(:`__Entity__`)<-[:MENTIONS]-(d:`__Chunk__`)
            WITH c, count(distinct d) AS rank
            SET c.community_rank = rank
            RETURN count(c) AS processed_count
            """)
            
            processed_count = result[0]['processed_count'] if result else 0
            elapsed_time = time.time() - start_time
            print(f"社区权重计算完成，处理了 {processed_count} 个社区，耗时: {elapsed_time:.2f}秒")
        except Exception as e:
            print(f"计算社区权重时出错: {e}")
            # 尝试使用更简单的查询
            try:
                print("尝试使用简化方法计算社区权重...")
                self.graph.query("""
                MATCH (c:`__Community__`)<-[:IN_COMMUNITY]-(e:`__Entity__`)
                WITH c, count(e) AS entity_count
                SET c.community_rank = entity_count
                """)
                print("使用实体计数作为社区权重")
            except Exception as e2:
                print(f"简化社区权重计算也失败: {e2}")
                print("继续处理，但社区将没有权重信息")

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
        if not summaries:
            print("没有社区摘要需要存储")
            return
            
        start_time = time.time()
        print(f"开始存储 {len(summaries)} 个社区摘要...")
        
        # 使用批处理优化存储性能
        batch_size = min(100, max(10, len(summaries) // 5))
        total_batches = (len(summaries) + batch_size - 1) // batch_size
        
        for i in range(0, len(summaries), batch_size):
            batch = summaries[i:i+batch_size]
            batch_start = time.time()
            
            try:
                self.graph.query("""
                UNWIND $data AS row
                MERGE (c:__Community__ {id:row.community})
                SET c.summary = row.summary, 
                    c.full_content = row.full_content,
                    c.summary_created_at = datetime()
                """, params={"data": batch})
                
                batch_time = time.time() - batch_start
                print(f"已存储批次 {i//batch_size + 1}/{total_batches}, "
                      f"耗时: {batch_time:.2f}秒")
                
            except Exception as e:
                print(f"存储社区摘要批次时出错: {e}")
                # 尝试逐个存储
                for summary in batch:
                    try:
                        self.graph.query("""
                        MERGE (c:__Community__ {id:$community})
                        SET c.summary = $summary, 
                            c.full_content = $full_content,
                            c.summary_created_at = datetime()
                        """, params=summary)
                    except Exception as e2:
                        print(f"存储单个社区摘要时出错: {e2}")
        
        total_time = time.time() - start_time
        print(f"社区摘要存储完成，总耗时: {total_time:.2f}秒")

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
        
        # 性能监控
        self.llm_time = 0
        self.query_time = 0
        self.store_time = 0
        
        # 设置并行处理的线程数
        self.max_workers = os.cpu_count() or 4
        print(f"社区摘要生成器初始化，并行线程数: {self.max_workers}")

    def _setup_llm_chain(self) -> None:
        """设置LLM处理链"""
        try:
            community_prompt = ChatPromptTemplate.from_messages([
                ("system", "给定一个输入三元组，生成信息摘要。没有序言。"),
                ("human", community_template),
            ])
            self.community_chain = community_prompt | self.llm | StrOutputParser()
        except Exception as e:
            print(f"设置LLM处理链时出错: {e}")
            raise

    def _process_community(self, community: Dict) -> Dict:
        """处理单个社区并生成摘要"""
        start_time = time.time()
        community_id = community.get('communityId', 'unknown')
        
        try:
            # 格式化社区信息
            stringify_info = self.describer.prepare_string(community)
            
            # 检查社区信息是否有效
            if len(stringify_info) < 10:
                print(f"社区 {community_id} 的信息太少，跳过摘要生成")
                return {
                    "community": community_id,
                    "summary": "此社区没有足够的信息生成摘要。",
                    "full_content": stringify_info
                }
            
            # 调用LLM生成摘要
            summary = self.community_chain.invoke({'community_info': stringify_info})
            
            elapsed_time = time.time() - start_time
            print(f"社区 {community_id} 摘要生成完成，耗时: {elapsed_time:.2f}秒")
            
            return {
                "community": community_id,
                "summary": summary,
                "full_content": stringify_info
            }
        except Exception as e:
            print(f"处理社区 {community_id} 摘要时出错: {e}")
            # 返回错误信息作为摘要
            return {
                "community": community_id,
                "summary": f"生成摘要时出错: {str(e)}",
                "full_content": community.get('full_content', str(community))
            }

    @abstractmethod
    def collect_community_info(self) -> List[Dict]:
        """收集社区信息的抽象方法"""
        pass

    def process_communities(self) -> List[Dict]:
        """处理所有社区的完整流程"""
        total_start_time = time.time()
        print("开始处理社区摘要...")
        
        try:
            # 计算社区权重
            rank_start = time.time()
            self.ranker.calculate_ranks()
            rank_time = time.time() - rank_start
            
            # 收集社区信息
            query_start = time.time()
            community_info = self.collect_community_info()
            self.query_time = time.time() - query_start
            
            community_count = len(community_info)
            print(f"收集到 {community_count} 个社区信息，耗时: {self.query_time:.2f}秒")
            
            # 如果没有社区，提前返回
            if not community_info:
                print("没有找到需要处理的社区")
                return []
            
            # 动态决定并行度
            optimal_workers = min(self.max_workers, max(1, community_count // 2))
            
            # 并行生成摘要
            llm_start = time.time()
            print(f"开始并行生成 {community_count} 个社区的摘要，使用 {optimal_workers} 个线程...")
            
            summaries = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=optimal_workers) as executor:
                # 提交所有任务
                future_to_community = {
                    executor.submit(self._process_community, info): i 
                    for i, info in enumerate(community_info)
                }
                
                # 处理完成的任务
                for i, future in enumerate(concurrent.futures.as_completed(future_to_community)):
                    try:
                        result = future.result()
                        summaries.append(result)
                        
                        # 定期报告进度
                        if (i+1) % 10 == 0 or (i+1) == community_count:
                            print(f"已处理 {i+1}/{community_count} 个社区摘要 "
                                  f"({(i+1)/community_count*100:.1f}%)")
                            
                    except Exception as e:
                        print(f"处理社区摘要时出错: {e}")
            
            self.llm_time = time.time() - llm_start
            print(f"社区摘要生成完成，耗时: {self.llm_time:.2f}秒")
            
            # 保存摘要
            store_start = time.time()
            self.storer.store_summaries(summaries)
            self.store_time = time.time() - store_start
            
            total_time = time.time() - total_start_time
            
            # 输出性能统计
            print(f"\n社区摘要处理完成，总耗时: {total_time:.2f}秒")
            print(f"  社区权重计算: {rank_time:.2f}秒 ({rank_time/total_time*100:.1f}%)")
            print(f"  社区信息查询: {self.query_time:.2f}秒 ({self.query_time/total_time*100:.1f}%)")
            print(f"  摘要生成(LLM): {self.llm_time:.2f}秒 ({self.llm_time/total_time*100:.1f}%)")
            print(f"  结果存储: {self.store_time:.2f}秒 ({self.store_time/total_time*100:.1f}%)")
            
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
        start_time = time.time()
        print("收集Leiden社区信息...")
        
        try:
            # 优化后的查询 - 先获取社区总数以确定处理方式
            count_query = """
            MATCH (c:`__Community__` {level: 0})
            RETURN count(c) AS community_count
            """
            
            count_result = self.graph.query(count_query)
            community_count = count_result[0]['community_count'] if count_result else 0
            
            if community_count == 0:
                print("没有找到Leiden社区")
                return []
                
            print(f"找到 {community_count} 个Leiden社区，开始收集详细信息")
            
            # 根据社区数量决定处理策略
            if community_count > 1000:
                print("社区数量较多，使用分批处理")
                return self._collect_info_in_batches(community_count)
            else:
                # 修复ORDER BY语法错误 - 移除NULLS LAST，改用CASE表达式处理空值
                result = self.graph.query("""
                // 找到最底层(level=0)的社区
                MATCH (c:`__Community__` {level: 0})
                // 优先处理有较高排名的社区 - 修复排序语法
                WITH c ORDER BY CASE WHEN c.community_rank IS NULL THEN 0 ELSE c.community_rank END DESC
                LIMIT 200  // 限制处理数量以提高性能
                
                MATCH (c)<-[:IN_COMMUNITY]-(e:__Entity__)
                WITH c, collect(e) as nodes
                WHERE size(nodes) > 1
                
                // 获取实体间的关系 - 使用参数化查询控制复杂度
                CALL {
                    WITH nodes
                    MATCH (n1:__Entity__)
                    WHERE n1 IN nodes
                    MATCH (n2:__Entity__)
                    WHERE n2 IN nodes AND id(n1) < id(n2)
                    MATCH (n1)-[r]->(n2)
                    RETURN collect(distinct r) as relationships
                }
                
                RETURN c.id AS communityId,
                    [n in nodes | {
                        id: n.id, 
                        description: n.description, 
                        type: CASE WHEN size([el in labels(n) WHERE el <> '__Entity__']) > 0 
                                THEN [el in labels(n) WHERE el <> '__Entity__'][0] 
                                ELSE 'Unknown' END
                    }] AS nodes,
                    [r in relationships | {
                        start: startNode(r).id, 
                        type: type(r), 
                        end: endNode(r).id, 
                        description: r.description
                    }] AS rels
                """)
                
                elapsed_time = time.time() - start_time
                print(f"收集到 {len(result)} 个Leiden社区信息，耗时: {elapsed_time:.2f}秒")
                return result
                
        except Exception as e:
            print(f"收集Leiden社区信息时出错: {e}")
            # 尝试使用更简单的查询
            try:
                print("尝试使用简化查询收集社区信息...")
                simplified_result = self.graph.query("""
                MATCH (c:`__Community__` {level: 0})
                WITH c LIMIT 50
                MATCH (c)<-[:IN_COMMUNITY]-(e:__Entity__)
                WITH c, collect(e) as nodes
                WHERE size(nodes) > 1
                RETURN c.id AS communityId,
                    [n in nodes | {
                        id: n.id, 
                        description: coalesce(n.description, 'No description'), 
                        type: CASE WHEN size(labels(n)) > 0 THEN labels(n)[0] ELSE 'Unknown' END
                    }] AS nodes,
                    [] AS rels
                """)
                
                print(f"使用简化查询收集到 {len(simplified_result)} 个社区信息")
                return simplified_result
            except Exception as e2:
                print(f"简化查询也失败: {e2}")
                print("返回空结果")
                return []
    
    def _collect_info_in_batches(self, total_count: int) -> List[Dict]:
        """分批收集社区信息，适用于大量社区的情况"""
        batch_size = 50
        total_batches = (total_count + batch_size - 1) // batch_size
        all_results = []
        
        print(f"使用批处理收集社区信息，共 {total_batches} 批")
        
        for batch in range(total_batches):
            if batch > 20:  # 限制批次数量以避免处理过长
                print(f"已达到最大批次限制(20)，停止收集")
                break
                
            skip = batch * batch_size
            
            try:
                # 修复ORDER BY语法错误 - 用CASE表达式代替NULLS LAST
                batch_result = self.graph.query("""
                // 找到最底层(level=0)的社区
                MATCH (c:`__Community__`)
                WHERE c.level = 0
                // 优先处理有较高排名的社区 - 修复排序语法
                WITH c ORDER BY CASE WHEN c.community_rank IS NULL THEN 0 ELSE c.community_rank END DESC
                SKIP $skip LIMIT $batch_size
                
                MATCH (c)<-[:IN_COMMUNITY]-(e:__Entity__)
                WITH c, collect(e) as nodes
                WHERE size(nodes) > 1
                
                // 简化关系查询以提高性能
                CALL {
                    WITH nodes
                    MATCH (n1:__Entity__)
                    WHERE n1 IN nodes
                    MATCH (n2:__Entity__)
                    WHERE n2 IN nodes AND id(n1) < id(n2)
                    MATCH (n1)-[r]->(n2)
                    WITH collect(distinct r) as relationships
                    LIMIT 100  // 限制关系数量
                    RETURN relationships
                }
                
                RETURN c.id AS communityId,
                    [n in nodes | {
                        id: n.id, 
                        description: n.description, 
                        type: CASE WHEN size([el in labels(n) WHERE el <> '__Entity__']) > 0 
                            THEN [el in labels(n) WHERE el <> '__Entity__'][0] 
                            ELSE 'Unknown' END
                    }] AS nodes,
                    [r in relationships | {
                        start: startNode(r).id, 
                        type: type(r), 
                        end: endNode(r).id, 
                        description: r.description
                    }] AS rels
                """, params={"skip": skip, "batch_size": batch_size})
                
                all_results.extend(batch_result)
                print(f"批次 {batch+1}/{total_batches} 完成，收集到 {len(batch_result)} 个社区")
                
            except Exception as e:
                print(f"批次 {batch+1} 处理出错: {e}")
        
        return all_results

class SLLPASummarizer(BaseSummarizer):
    """
    SLLPA算法的社区摘要生成器。
    SLLPA算法生成的社区没有层级结构，所有社区都在同一层级(level=0)。
    """
    
    def collect_community_info(self) -> List[Dict]:
        """收集SLLPA算法生成的社区信息"""
        start_time = time.time()
        print("收集SLLPA社区信息...")
        
        try:
            # 优化后的查询 - 先获取社区总数以确定处理方式
            count_query = """
            MATCH (c:`__Community__`)
            WHERE c.level = 0  // SLLPA的所有社区都在level 0
            RETURN count(c) AS community_count
            """
            
            count_result = self.graph.query(count_query)
            community_count = count_result[0]['community_count'] if count_result else 0
            
            if community_count == 0:
                print("没有找到SLLPA社区")
                return []
                
            print(f"找到 {community_count} 个SLLPA社区，开始收集详细信息")
            
            # 根据社区数量决定处理策略
            if community_count > 1000:
                print("社区数量较多，使用分批处理")
                return self._collect_info_in_batches(community_count)
            else:
                # 优化后的查询 - 使用多阶段处理提高性能
                result = self.graph.query("""
                MATCH (c:`__Community__`)
                WHERE c.level = 0  // SLLPA的所有社区都在level 0
                // 优先处理有较高排名的社区
                WITH c ORDER BY c.community_rank DESC NULLS LAST
                LIMIT 200  // 限制处理数量以提高性能
                
                MATCH (c)<-[:IN_COMMUNITY*]-(e:__Entity__)
                WITH c, collect(e) AS nodes
                WHERE size(nodes) > 1  // 只选择包含多个节点的社区
                
                // 使用存储过程安全地获取子图
                CALL {
                    WITH nodes
                    MATCH (n1:__Entity__)
                    WHERE n1 IN nodes
                    MATCH (n2:__Entity__)
                    WHERE n2 IN nodes AND id(n1) < id(n2)
                    MATCH (n1)-[r]->(n2)
                    WITH collect(distinct r) as relationships
                    LIMIT 100  // 限制关系数量
                    RETURN relationships
                }
                
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
                
                elapsed_time = time.time() - start_time
                print(f"收集到 {len(result)} 个SLLPA社区信息，耗时: {elapsed_time:.2f}秒")
                return result
                
        except Exception as e:
            print(f"收集SLLPA社区信息时出错: {e}")
            # 尝试使用更简单的查询
            try:
                print("尝试使用简化查询收集社区信息...")
                simplified_result = self.graph.query("""
                MATCH (c:`__Community__`)
                WHERE c.level = 0
                WITH c LIMIT 50
                MATCH (c)<-[:IN_COMMUNITY]-(e:__Entity__)
                WITH c, collect(e) as nodes
                WHERE size(nodes) > 1
                RETURN c.id AS communityId,
                       [n in nodes | {
                           id: n.id, 
                           description: coalesce(n.description, 'No description'), 
                           type: labels(n)[0]
                       }] AS nodes,
                       [] AS rels
                """)
                
                print(f"使用简化查询收集到 {len(simplified_result)} 个社区信息")
                return simplified_result
            except Exception as e2:
                print(f"简化查询也失败: {e2}")
                print("返回空结果")
                return []
    
    def _collect_info_in_batches(self, total_count: int) -> List[Dict]:
        """分批收集社区信息，适用于大量社区的情况"""
        batch_size = 50
        total_batches = (total_count + batch_size - 1) // batch_size
        all_results = []
        
        print(f"使用批处理收集SLLPA社区信息，共 {total_batches} 批")
        
        for batch in range(total_batches):
            if batch > 20:  # 限制批次数量以避免处理过长
                print(f"已达到最大批次限制(20)，停止收集")
                break
                
            skip = batch * batch_size
            
            try:
                batch_result = self.graph.query(f"""
                MATCH (c:`__Community__`)
                WHERE c.level = 0
                // 优先处理有较高排名的社区
                WITH c ORDER BY c.community_rank DESC NULLS LAST
                SKIP {skip} LIMIT {batch_size}
                
                MATCH (c)<-[:IN_COMMUNITY]-(e:__Entity__)
                WITH c, collect(e) as nodes
                WHERE size(nodes) > 1
                
                // 简化关系查询以提高性能
                CALL {{
                    WITH nodes
                    WITH nodes[0..20] AS limited_nodes  // 限制节点数量
                    MATCH (n1)-->(n2)
                    WHERE n1 IN limited_nodes AND n2 IN limited_nodes
                    RETURN collect(distinct relationship(n1, n2)) as relationships
                }}
                
                RETURN c.id AS communityId,
                       [n in nodes | {{
                           id: n.id, 
                           description: n.description, 
                           type: [el in labels(n) WHERE el <> '__Entity__'][0]
                       }}] AS nodes,
                       [r in relationships | {{
                           start: startNode(r).id, 
                           type: type(r), 
                           end: endNode(r).id, 
                           description: r.description
                       }}] AS rels
                """)
                
                all_results.extend(batch_result)
                print(f"批次 {batch+1}/{total_batches} 完成，收集到 {len(batch_result)} 个社区")
                
            except Exception as e:
                print(f"批次 {batch+1} 处理出错: {e}")
        
        return all_results

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
        print(f"创建 {algorithm} 算法的社区摘要生成器...")
        
        if algorithm.lower() == 'leiden':
            return LeidenSummarizer(graph)
        elif algorithm.lower() == 'sllpa':
            return SLLPASummarizer(graph)
        else:
            error_msg = f"不支持的算法类型: {algorithm}"
            print(f"错误: {error_msg}")
            raise ValueError(error_msg)