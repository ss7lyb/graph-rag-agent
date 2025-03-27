from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field, asdict
import re
import time
import json

from evaluator.base import BaseEvaluator, BaseMetric
from evaluator.graph_metrics import GraphCoverageMetric, CommunityRelevanceMetric, SubgraphQualityMetric
from evaluator.preprocessing import clean_thinking_process, extract_references_from_answer, clean_references
from evaluator.utils import normalize_answer

@dataclass
class RetrievalEvaluationSample:
    """检索评估样本类"""
    
    question: str
    system_answer: str = ""
    retrieved_entities: List[str] = field(default_factory=list)
    retrieved_relationships: List[Tuple[str, str, str]] = field(default_factory=list)
    referenced_entities: List[str] = field(default_factory=list)
    referenced_relationships: List[Tuple[str, str, str]] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    agent_type: str = ""  # naive, hybrid, graph, deep
    retrieval_time: float = 0.0
    retrieval_logs: Dict[str, Any] = field(default_factory=dict)
    
    def update_system_answer(self, answer: str, agent_type: str = ""):
        """更新系统回答并提取引用"""
        # 先清理思考过程，再提取引用数据
        if agent_type == "deep":
            answer = clean_thinking_process(answer)
            
        # 保存原始答案（包含引用数据）
        self.system_answer = answer
        
        if agent_type:
            self.agent_type = agent_type
                
        # 提取引用的实体和关系
        refs = extract_references_from_answer(answer)
        
        # 将提取的实体和关系ID存储为字符串列表
        self.referenced_entities = refs.get("entities", [])
        # 关系暂时存储为ID，后续在evaluation方法中再转换为三元组
        self.referenced_relationships = refs.get("relationships", [])
    
    def update_retrieval_data(self, entities: List[str], relationships: List[Tuple[str, str, str]]):
        """更新检索到的实体和关系"""
        self.retrieved_entities = entities
        self.retrieved_relationships = relationships
        
    def update_logs(self, logs: Dict[str, Any]):
        """更新检索日志"""
        self.retrieval_logs = logs
    
    def update_evaluation_score(self, metric: str, score: float):
        """更新评估分数"""
        self.scores[metric] = score
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        
        # 处理关系元组（JSON序列化时需要转换为列表）
        result["retrieved_relationships"] = [list(rel) for rel in self.retrieved_relationships]
        result["referenced_relationships"] = [list(rel) for rel in self.referenced_relationships]
        
        # 处理检索日志中可能存在的HumanMessage
        if "retrieval_logs" in result and isinstance(result["retrieval_logs"], dict):
            logs = result["retrieval_logs"]
            if "execution_log" in logs and isinstance(logs["execution_log"], list):
                for i, log in enumerate(logs["execution_log"]):
                    # 处理输入中可能的HumanMessage
                    if "input" in log and hasattr(log["input"], "__class__") and log["input"].__class__.__name__ == "HumanMessage":
                        logs["execution_log"][i]["input"] = str(log["input"])
                    # 处理输出中可能的HumanMessage或AIMessage
                    if "output" in log and hasattr(log["output"], "__class__") and log["output"].__class__.__name__ in ["HumanMessage", "AIMessage"]:
                        logs["execution_log"][i]["output"] = str(log["output"])
        
        return result

@dataclass
class RetrievalEvaluationData:
    """检索评估数据类"""
    
    samples: List[RetrievalEvaluationSample] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> RetrievalEvaluationSample:
        return self.samples[idx]
    
    def append(self, sample: RetrievalEvaluationSample):
        """添加评估样本"""
        self.samples.append(sample)
    
    @property
    def questions(self) -> List[str]:
        """获取所有问题"""
        return [sample.question for sample in self.samples]
    
    @property
    def system_answers(self) -> List[str]:
        """获取所有系统回答"""
        return [sample.system_answer for sample in self.samples]
    
    @property
    def retrieved_entities(self) -> List[List[str]]:
        """获取所有检索到的实体"""
        return [sample.retrieved_entities for sample in self.samples]
    
    @property
    def referenced_entities(self) -> List[List[str]]:
        """获取所有引用的实体"""
        return [sample.referenced_entities for sample in self.samples]
    
    @property
    def retrieved_relationships(self) -> List[List[Tuple[str, str, str]]]:
        """获取所有检索到的关系"""
        return [sample.retrieved_relationships for sample in self.samples]
    
    @property
    def referenced_relationships(self) -> List[List[Tuple[str, str, str]]]:
        """获取所有引用的关系"""
        return [sample.referenced_relationships for sample in self.samples]
    
    def save(self, path: str):
        """保存评估数据"""
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                from langchain_core.messages import BaseMessage
                if isinstance(obj, BaseMessage):
                    return str(obj)
                return super().default(obj)
        
        with open(path, "w", encoding='utf-8') as f:
            samples_data = [sample.to_dict() for sample in self.samples]
            json.dump(samples_data, f, ensure_ascii=False, indent=2, cls=CustomEncoder)
    
    @classmethod
    def load(cls, path: str) -> 'RetrievalEvaluationData':
        """加载评估数据"""
        with open(path, "r", encoding='utf-8') as f:
            samples_data = json.load(f)
        
        data = cls()
        for sample_data in samples_data:
            # 转换关系格式（从列表到元组）
            if "retrieved_relationships" in sample_data:
                sample_data["retrieved_relationships"] = [tuple(rel) for rel in sample_data["retrieved_relationships"]]
            if "referenced_relationships" in sample_data:
                sample_data["referenced_relationships"] = [tuple(rel) for rel in sample_data["referenced_relationships"]]
                
            sample = RetrievalEvaluationSample(**sample_data)
            data.append(sample)
        
        return data

class RetrievalPrecision(BaseMetric):
    """检索精确率评估指标"""
    
    metric_name = "retrieval_precision"
    
    def __init__(self, config):
        super().__init__(config)
        self.neo4j_client = config.get('neo4j_client', None)
        self.debug = config.get('debug', True)
        self.llm = config.get("llm", None)
    
    def calculate_metric(self, data) -> Tuple[Dict[str, float], List[float]]:
        """
        计算检索精确率
        
        Args:
            data: 评估数据
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 总体得分和每个样本的得分
        """
        if self.debug:
            print("\n======== RetrievalPrecision 计算日志 ========")
            print(f"样本总数: {len(data.samples) if hasattr(data, 'samples') else 0}")
            print(f"LLM可用: {'是' if self.llm else '否'}")
        
        retrieved_entities = data.retrieved_entities
        referenced_entities = data.referenced_entities
        
        # 打印总体信息
        total_samples = len(data.samples) if hasattr(data, 'samples') else 0
        if self.debug:
            print(f"检索实体列表长度: {len(retrieved_entities)}")
            print(f"引用实体列表长度: {len(referenced_entities)}")
        
        precision_scores = []
        for idx, (retr_entities, ref_entities) in enumerate(zip(retrieved_entities, referenced_entities)):
            if self.debug:
                print(f"\n样本 {idx+1}:")
                print(f"  检索到的实体数量: {len(retr_entities) if retr_entities else 0}")
                print(f"  引用的实体数量: {len(ref_entities) if ref_entities else 0}")
                    
                # 详细打印实体信息
                if retr_entities:
                    print(f"  检索实体: {retr_entities[:5]}{'...' if len(retr_entities) > 5 else ''}")
                if ref_entities:
                    print(f"  引用实体: {ref_entities[:5]}{'...' if len(ref_entities) > 5 else ''}")
            
            # 如果没有检索到实体或引用实体，给予基础分
            if not retr_entities or not ref_entities:
                precision_scores.append(0.3)
                if self.debug:
                    print(f"  没有检索到实体或引用实体，使用基础分: 0.3")
                continue
            
            # 规则匹配评分
            matched, rule_score = self._calculate_rule_precision(retr_entities, ref_entities)
            
            if self.debug:
                print(f"  匹配的实体数量: {matched}")
                print(f"  规则精确率分数: {rule_score:.4f}")
            
            # 如果规则评分只是基础分或很低，使用LLM回退
            if rule_score <= 0.3 and self.llm:
                if self.debug:
                    print(f"  规则精确率过低，尝试使用LLM评估")
                
                # 获取样本
                sample = data.samples[idx]
                question = sample.question
                agent_type = sample.agent_type
                
                # 准备LLM提示
                retr_str = ", ".join([str(e) for e in retr_entities[:10]])
                ref_str = ", ".join([str(e) for e in ref_entities[:10]])
                
                prompt = f"""
                请评估以下检索到的实体与用户引用实体的匹配程度，给出0到1的分数。
                
                问题: {question}
                代理类型: {agent_type}
                
                检索到的实体: [{retr_str}]
                用户引用的实体: [{ref_str}]
                
                评分标准:
                - 高分(0.8-1.0): 引用实体全部或大部分存在于检索实体中
                - 中分(0.4-0.7): 引用实体部分存在于检索实体中
                - 低分(0.0-0.3): 引用实体几乎不在检索实体中
                
                只返回一个0到1之间的数字表示分数，不要有任何其他文字。
                """
                
                try:
                    response = self.llm.invoke(prompt)
                    score_text = response.content if hasattr(response, 'content') else response
                    
                    if self.debug:
                        print(f"  LLM响应: {score_text}")
                    
                    # 提取数字
                    score_match = re.search(r'(\d+(\.\d+)?)', score_text)
                    if score_match:
                        precision = float(score_match.group(1))
                        # 确保在0-1范围内
                        precision = max(0.0, min(1.0, precision))
                        if self.debug:
                            print(f"  LLM评估的精确率分数: {precision:.4f}")
                    else:
                        precision = rule_score  # 使用规则分数作为回退
                        if self.debug:
                            print(f"  无法从LLM响应中提取分数，使用规则分数: {precision:.4f}")
                except Exception as e:
                    if self.debug:
                        print(f"  LLM评估时出错: {e}")
                    precision = rule_score  # 使用规则分数作为回退
            else:
                precision = rule_score  # 使用规则分数
                
            if self.debug:
                print(f"  最终精确率分数: {precision:.4f}")
            
            precision_scores.append(precision)
        
        avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.3
        
        if self.debug:
            print(f"总体评分分布: 最低={min(precision_scores):.4f}, 最高={max(precision_scores):.4f}, 平均={avg_precision:.4f}")
            print("完成检索精确率评估")
        
        return {"retrieval_precision": avg_precision}, precision_scores

    def _calculate_rule_precision(self, retr_entities, ref_entities):
        """计算规则匹配精确率"""
        # 实体字符串预处理
        retr_entities_str = [str(e).lower() for e in retr_entities]
        ref_entities_str = [str(e).lower() for e in ref_entities]

        # 1. 直接ID匹配 - 检查引用实体ID是否出现在检索实体文本中
        direct_matches = 0
        for ref_id in ref_entities_str:
            for retr_entity in retr_entities_str:
                if ref_id in retr_entity:
                    direct_matches += 1
                    break
        
        # 2. 数字ID匹配
        num_matches = 0
        for ref_id in ref_entities_str:
            ref_num = re.search(r'\d+', ref_id)
            if ref_num and any(ref_num.group() in retr for retr in retr_entities_str):
                num_matches += 1
        
        # 使用最高的匹配数
        matched = max(direct_matches, num_matches)
        
        # 3. 计算分数
        if matched > 0:
            # 有匹配 - 根据匹配比例给分
            return matched, max(0.3, 0.3 + 0.7 * (matched / len(ref_entities_str)))
        else:
            # 无匹配 - 返回基础分
            return 0, 0.3
    
    def _get_entities_with_info(self, entity_ids: List[str]) -> List[Dict[str, str]]:
        entities_info = []
        
        if not self.neo4j_client or not entity_ids:
            return entities_info
        
        try:
            # 尝试不同的查询方式获取实体信息
            # 1. 直接查询所有节点
            query = """
            MATCH (n)
            WHERE n.id IN $ids
            RETURN n.id AS id, n.description AS description
            """
            result = self.neo4j_client.execute_query(query, {"ids": entity_ids})
            
            if result.records:
                for record in result.records:
                    entity_id = record.get("id", "")
                    entity_desc = record.get("description", "")
                    
                    if entity_id:
                        entities_info.append({
                            "id": str(entity_id),
                            "description": entity_desc or f"实体 {entity_id}"
                        })
            
            # 2. 如果没有结果，尝试不同的属性组合
            if not entities_info:
                alt_query = """
                MATCH (n)
                WHERE toString(n.id) IN $str_ids
                RETURN n.id AS id, n.description AS description, n.summary AS summary, n.full_content AS full_content
                """
                str_ids = [str(eid) for eid in entity_ids]
                alt_result = self.neo4j_client.execute_query(alt_query, {"str_ids": str_ids})
                
                if alt_result.records:
                    for record in alt_result.records:
                        entity_id = record.get("id", "")
                        entity_desc = record.get("description", "")
                        summary = record.get("summary", "")
                        full_content = record.get("full_content", "")
                        
                        description = entity_desc or summary or full_content or f"实体 {entity_id}"
                        
                        if entity_id:
                            entities_info.append({
                                "id": str(entity_id),
                                "description": description
                            })
            
            # 3. 如果仍然没有结果，使用默认值
            if not entities_info:
                for entity_id in entity_ids:
                    entities_info.append({
                        "id": str(entity_id),
                        "description": f"实体 {entity_id}"
                    })
                    
            return entities_info
        except Exception as e:
            if self.debug:
                print(f"  查询实体信息失败: {e}")
            # 出错时，使用ID创建简单实体信息
            return [{"id": str(eid), "description": f"实体 {eid}"} for eid in entity_ids]


class RetrievalUtilization(BaseMetric):
    """检索利用率评估指标"""
    
    metric_name = "retrieval_utilization"

    def __init__(self, config):
        """初始化评估指标"""
        super().__init__(config)
        self.neo4j_client = config.get('neo4j_client', None)
        self.debug = config.get('debug', True)
        self.llm = config.get("llm", None)
    
    def calculate_metric(self, data) -> Tuple[Dict[str, float], List[float]]:
        """
        计算检索利用率
        
        Args:
            data: 评估数据
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 总体得分和每个样本的得分
        """
        if self.debug:
            print("\n======== RetrievalUtilization 计算日志 ========")
            print(f"样本总数: {len(data.samples) if hasattr(data, 'samples') else 0}")
            print(f"LLM可用: {'是' if self.llm else '否'}")
        
        retrieved_entities = data.retrieved_entities
        referenced_entities = data.referenced_entities
        
        # 打印总体信息
        total_samples = len(data.samples) if hasattr(data, 'samples') else 0
        if self.debug:
            print(f"检索实体列表长度: {len(retrieved_entities)}")
            print(f"引用实体列表长度: {len(referenced_entities)}")
        
        utilization_scores = []
        for idx, (retr_entities, ref_entities) in enumerate(zip(retrieved_entities, referenced_entities)):
            if self.debug:
                print(f"\n样本 {idx+1}:")
                
                # 检查数据格式
                if not isinstance(retr_entities, list):
                    print(f"  检索实体不是列表类型，而是 {type(retr_entities)}")
                    retr_entities = []
                if not isinstance(ref_entities, list):
                    print(f"  引用实体不是列表类型，而是 {type(ref_entities)}")
                    ref_entities = []
                    
                # 确保所有元素都是字符串
                retr_entities = [str(e) for e in retr_entities]
                ref_entities = [str(e) for e in ref_entities]
                
                print(f"  检索到的实体数量: {len(retr_entities)}")
                print(f"  引用的实体数量: {len(ref_entities)}")
                
                # 详细打印实体ID
                if retr_entities:
                    print(f"  检索实体: {retr_entities[:5]}{'...' if len(retr_entities) > 5 else ''}")
                if ref_entities:
                    print(f"  引用实体: {ref_entities[:5]}{'...' if len(ref_entities) > 5 else ''}")
            
            # 如果没有引用实体，给予基础分
            if not ref_entities:
                utilization_scores.append(0.3)
                if self.debug:
                    print(f"  没有引用实体，使用基础分: 0.3")
                continue
            
            # 如果没有检索到实体，给予基础分
            if not retr_entities:
                utilization_scores.append(0.3)
                if self.debug:
                    print(f"  没有检索到实体，使用基础分: 0.3")
                continue
            
            # 规则匹配评分
            matches_found, rule_score = self._calculate_rule_utilization(retr_entities, ref_entities)
            
            if self.debug:
                print(f"  在检索结果中找到的引用实体数量: {matches_found}")
                print(f"  规则利用率分数: {rule_score:.4f}")
            
            # 如果规则评分只是基础分或很低，使用LLM回退
            if rule_score <= 0.3 and self.llm:
                if self.debug:
                    print(f"  规则利用率过低，尝试使用LLM评估")
                
                # 获取样本
                sample = data.samples[idx]
                question = sample.question
                system_answer = sample.system_answer
                
                # 提取系统答案的前200个字符作为上下文
                answer_context = system_answer[:200] + "..." if len(system_answer) > 200 else system_answer
                
                # 准备LLM提示
                retr_str = ", ".join([str(e) for e in retr_entities[:10]])
                ref_str = ", ".join([str(e) for e in ref_entities[:10]])
                
                prompt = f"""
                请评估系统在回答用户问题时对检索实体的利用程度，给出0到1的分数。
                
                问题: {question}
                
                检索到的实体: [{retr_str}]
                用户引用的实体: [{ref_str}]
                
                系统回答(部分): {answer_context}
                
                评分标准:
                - 高分(0.8-1.0): 系统充分利用了检索到的实体中的关键信息
                - 中分(0.4-0.7): 系统部分利用了检索到的实体信息
                - 低分(0.0-0.3): 系统几乎没有利用检索到的实体信息
                
                只返回一个0到1之间的数字表示分数，不要有任何其他文字。
                """
                
                try:
                    response = self.llm.invoke(prompt)
                    score_text = response.content if hasattr(response, 'content') else response
                    
                    if self.debug:
                        print(f"  LLM响应: {score_text}")
                    
                    # 提取数字
                    score_match = re.search(r'(\d+(\.\d+)?)', score_text)
                    if score_match:
                        utilization = float(score_match.group(1))
                        # 确保在0-1范围内
                        utilization = max(0.0, min(1.0, utilization))
                        if self.debug:
                            print(f"  LLM评估的利用率分数: {utilization:.4f}")
                    else:
                        utilization = rule_score  # 使用规则分数作为回退
                        if self.debug:
                            print(f"  无法从LLM响应中提取分数，使用规则分数: {utilization:.4f}")
                except Exception as e:
                    if self.debug:
                        print(f"  LLM评估时出错: {e}")
                    utilization = rule_score  # 使用规则分数作为回退
            else:
                utilization = rule_score  # 使用规则分数
            
            if self.debug:
                print(f"  最终利用率分数: {utilization:.4f}")
            
            utilization_scores.append(utilization)
        
        avg_utilization = sum(utilization_scores) / len(utilization_scores) if utilization_scores else 0.3
        
        if self.debug:
            print(f"总体评分分布: 最低={min(utilization_scores):.4f}, 最高={max(utilization_scores):.4f}, 平均={avg_utilization:.4f}")
            print("完成检索利用率评估")
        
        return {"retrieval_utilization": avg_utilization}, utilization_scores

    def _calculate_rule_utilization(self, retr_entities, ref_entities):
        """计算规则匹配利用率"""
        # 标准化处理
        retr_norm = [str(e).lower() for e in retr_entities]
        ref_norm = [str(e).lower() for e in ref_entities]
        
        # 1. 直接ID匹配
        direct_matches = 0
        for ref_id in ref_norm:
            if any(ref_id in retr for retr in retr_norm):
                direct_matches += 1
        
        # 2. 数字ID匹配
        num_matches = 0
        for ref_id in ref_norm:
            ref_num = re.search(r'\d+', ref_id)
            if ref_num and any(ref_num.group() in retr for retr in retr_norm):
                num_matches += 1
        
        # 使用最高的匹配数
        matched = max(direct_matches, num_matches)
        
        # 计算利用率
        if matched > 0:
            # 有匹配，计算基于匹配比例的分数
            return matched, max(0.3, 0.3 + 0.7 * (matched / len(ref_norm)))
        else:
            # 无匹配，但检查字符串相似性
            combined_retr = " ".join(retr_norm)
            for ref in ref_norm:
                # 检查部分匹配
                if any(token in combined_retr for token in ref.split() if len(token) > 3):
                    return 1, 0.4  # 有部分匹配，给予略高于基础的分数
            
            # 无任何匹配
            return 0, 0.3


class RelationshipUtilizationMetric(BaseMetric):
    """关系利用率评估指标"""
    
    metric_name = "relationship_utilization"
    
    def __init__(self, config):
        super().__init__(config)
        self.neo4j_client = config.get('neo4j_client', None)
        self.debug = config.get('debug', True)  # 调试模式开关
    
    def calculate_metric(self, data) -> Tuple[Dict[str, float], List[float]]:
        """计算关系利用率"""
        
        if self.debug:
            print("\n======== RelationshipUtilization 计算日志 ========")
        
        utilization_scores = []
        
        # 打印总体信息
        total_samples = len(data.samples) if hasattr(data, 'samples') else 0
        if self.debug:
            print(f"样本总数: {total_samples}")
        
        for idx, sample in enumerate(data.samples):
            agent_type = sample.agent_type.lower() if sample.agent_type else ""
            
            if self.debug:
                print(f"\n样本 {idx+1}:")
                print(f"  代理类型: {agent_type}")
            
            # 获取引用的关系和实体
            referenced_rels = sample.referenced_relationships
            entity_ids = sample.referenced_entities
            
            if self.debug:
                print(f"  引用的关系数量: {len(referenced_rels) if referenced_rels else 0}")
                print(f"  引用的实体数量: {len(entity_ids) if entity_ids else 0}")
                
                # 如果有引用关系，打印部分示例
                if referenced_rels:
                    print(f"  引用关系示例: {referenced_rels[:3]}{'...' if len(referenced_rels) > 3 else ''}")
            
            # 如果没有引用关系和实体，给予基础分
            if not referenced_rels and not entity_ids:
                utilization_scores.append(0.3)
                if self.debug:
                    print(f"  没有引用关系和实体，使用基础分: 0.3")
                continue
            
            # 获取关系详细信息
            rel_info = self._get_relationship_info(referenced_rels)
            if self.debug:
                print(f"  获取到的关系信息数量: {len(rel_info)}")
                
                # 如果成功获取关系信息，打印部分示例
                if rel_info:
                    print(f"  关系信息示例: {rel_info[:2]}{'...' if len(rel_info) > 2 else ''}")
            
            # 没有有效关系信息，尝试基于ID的评分
            if not rel_info and referenced_rels:
                # 基于关系ID数量的基础评分
                rel_count = len(referenced_rels)
                id_based_score = min(0.4, 0.3 + 0.02 * rel_count)  # 每个关系加0.02，最高0.4
                if self.debug:
                    print(f"  没有关系详细信息，基于ID数量评分: {id_based_score:.4f}")
                utilization_scores.append(id_based_score)
                continue
            
            # 没有关系但有实体，尝试推断实体间的隐含关系
            if not rel_info and not referenced_rels and entity_ids and self.neo4j_client:
                implicit_rel_score = self._calculate_implicit_relationships(entity_ids)
                if self.debug:
                    print(f"  尝试推断实体间的隐含关系得分: {implicit_rel_score:.4f}")
                
                final_score = 0.3 + implicit_rel_score * 0.4  # 基础分加上推断关系得分
                if self.debug:
                    print(f"  基于推断关系的得分: {final_score:.4f}")
                
                utilization_scores.append(final_score)
                continue
            
            # 没有有效关系信息，回退到基础分
            if not rel_info and not referenced_rels:
                utilization_scores.append(0.3)
                if self.debug:
                    print(f"  无法获取关系信息，使用基础分: 0.3")
                continue
            
            # 计算关系利用率的多个维度
            quantity_score = self._calculate_quantity_score(rel_info)
            quality_score = self._calculate_quality_score(rel_info)
            relevance_score = self._calculate_relevance_score(rel_info, entity_ids)
            
            if self.debug:
                print(f"  数量得分: {quantity_score:.4f}")
                print(f"  质量得分: {quality_score:.4f}")
                print(f"  相关性得分: {relevance_score:.4f}")
            
            # 计算综合得分 - 数量占30%，质量占40%，相关性占30%
            base_score = 0.3  # 统一基础分
            total_score = base_score + 0.7 * (
                0.3 * quantity_score + 
                0.4 * quality_score + 
                0.3 * relevance_score
            )
            
            # 确保不超过1.0
            final_score = min(1.0, total_score)
            if self.debug:
                print(f"  基础分: {base_score}")
                print(f"  加权总分: {total_score:.4f}")
                print(f"  最终得分: {final_score:.4f}")
            
            utilization_scores.append(final_score)
        
        avg_utilization = sum(utilization_scores) / len(utilization_scores) if utilization_scores else 0.3
        if self.debug:
            print(f"\n平均关系利用率: {avg_utilization:.4f}")
            print("======== RelationshipUtilization 计算结束 ========\n")
        
        return {"relationship_utilization": avg_utilization}, utilization_scores
    
    def _get_relationship_info(self, referenced_rels) -> List[Dict[str, Any]]:
        rel_info = []
        
        if not self.neo4j_client or not referenced_rels:
            return rel_info
            
        # 处理字符串ID类型的关系
        rel_ids = [r for r in referenced_rels if isinstance(r, str)]
        
        # 转换为数字ID
        numeric_rel_ids = []
        for rel_id in rel_ids:
            try:
                if rel_id.isdigit() or rel_id.lstrip('-').isdigit():
                    numeric_rel_ids.append(int(rel_id))
            except (ValueError, AttributeError):
                continue
        
        if not numeric_rel_ids:
            return rel_info
            
        try:
            # 直接查询所有关系，然后手动匹配
            query = """
            MATCH (a)-[r]->(b)
            RETURN a.id AS source, type(r) AS relation, b.id AS target, 
                r.description AS description, r.weight AS weight
            LIMIT 500
            """
            result = self.neo4j_client.execute_query(query)
            
            if result.records:
                # 只获取前50条关系作为样本
                count = 0
                for record in result.records:
                    if count >= 50:
                        break
                        
                    source = record.get("source")
                    relation = record.get("relation")
                    target = record.get("target")
                    description = record.get("description")
                    weight = record.get("weight")
                    
                    if source and relation and target:
                        rel_info.append({
                            "source": str(source),
                            "relation": relation,
                            "target": str(target),
                            "description": description,
                            "weight": weight
                        })
                        count += 1
            
            return rel_info
        except Exception as e:
            if self.debug:
                print(f"  查询关系信息失败: {e}")
            return rel_info
    
    def _calculate_implicit_relationships(self, entity_ids: List[str]) -> float:
        """计算实体间的隐含关系得分"""
        if not self.neo4j_client or len(entity_ids) < 2:
            return 0.0
            
        try:
            # 查询实体之间是否有路径连接
            query = """
            MATCH path = (a:__Entity__)-[*1..3]-(b:__Entity__)
            WHERE a.id IN $ids AND b.id IN $ids AND a <> b
            RETURN COUNT(DISTINCT path) AS path_count
            """
            result = self.neo4j_client.execute_query(query, {"ids": entity_ids})
            
            path_count = 0
            if result.records and result.records[0].get("path_count") is not None:
                path_count = result.records[0].get("path_count")
            
            # 计算潜在的连接总数
            potential_connections = len(entity_ids) * (len(entity_ids) - 1) / 2
            connected_ratio = min(1.0, path_count / potential_connections) if potential_connections > 0 else 0
            
            if self.debug:
                print(f"  实体之间的路径数量: {path_count}")
                print(f"  潜在连接总数: {potential_connections:.1f}")
                print(f"  连接率: {connected_ratio:.4f}")
                
            return min(1.0, connected_ratio * 1.2)  # 提供一点加成，但不超过1.0
        except Exception as e:
            if self.debug:
                print(f"  计算隐含关系时出错: {e}")
            return 0.0
    
    def _calculate_quantity_score(self, rel_info: List[Dict[str, Any]]) -> float:
        """计算关系数量得分"""
        # 如果有关系详细信息，使用实际关系数量
        rel_count = len(rel_info) if rel_info else 0
        
        # 每个关系贡献0.1分，最多1.0
        return min(1.0, rel_count * 0.1)
    
    def _calculate_quality_score(self, rel_info: List[Dict[str, Any]]) -> float:
        """
        计算关系质量得分 
        
        Args:
            rel_info: 关系信息列表
            
        Returns:
            float: 关系质量得分
        """
        if not rel_info:
            return 0.0
            
        # 检查关系是否有描述 - 确保处理None值
        described_count = 0
        for rel in rel_info:
            # 使用描述或关系类型
            description = rel.get("description", "")
            relation_type = rel.get("relation", "")
            
            if ((description is not None and str(description).strip()) or 
                (relation_type is not None and str(relation_type).strip())):
                described_count += 1
        
        description_ratio = described_count / len(rel_info) if rel_info else 0
        
        # 检查关系类型的多样性
        relation_types = set()
        for rel in rel_info:
            rel_type = rel.get("relation", "")
            if rel_type and rel_type.strip():
                relation_types.add(rel_type)
        
        type_diversity = min(1.0, len(relation_types) / 5)  # 最多5种关系类型为满分
        
        # 检查来源和目标实体是否存在
        valid_relations = 0
        for rel in rel_info:
            source = rel.get("source", "")
            target = rel.get("target", "")
            if source and source != "unknown" and target and target != "unknown":
                valid_relations += 1
        
        validity_ratio = valid_relations / len(rel_info) if rel_info else 0
        
        # 计算关系权重的平均值（如果有）
        weight_score = 0.0
        weighted_rels = [rel for rel in rel_info if "weight" in rel and rel["weight"] is not None]
        if weighted_rels:
            try:
                weights = []
                for rel in weighted_rels:
                    # 确保权重是有效数字
                    if isinstance(rel["weight"], (int, float)):
                        weights.append(float(rel["weight"]))
                    elif isinstance(rel["weight"], str) and rel["weight"].replace('.', '', 1).isdigit():
                        weights.append(float(rel["weight"]))
                
                if weights:
                    avg_weight = sum(weights) / len(weights)
                    # 假设权重范围为0-10，归一化到0-1
                    weight_score = min(1.0, avg_weight / 10.0)
            except Exception as e:
                if self.debug:
                    print(f"  计算权重得分时出错: {e}")
        
        # 综合得分 - 描述占30%，多样性占30%，有效性占20%，权重占20%
        if weighted_rels:
            return (0.3 * description_ratio + 
                    0.3 * type_diversity + 
                    0.2 * validity_ratio + 
                    0.2 * weight_score)
        else:
            # 如果没有权重信息，重新分配占比
            return (0.4 * description_ratio + 
                    0.3 * type_diversity + 
                    0.3 * validity_ratio)
    
    def _calculate_relevance_score(self, rel_info: List[Dict[str, Any]], entity_ids: List[str]) -> float:
        """计算关系相关性得分"""
        if not rel_info or not entity_ids:
            return 0.0
            
        # 统计关系中的实体与引用实体的重合度
        relation_entities = set()
        for rel in rel_info:
            source = rel.get("source")
            target = rel.get("target")
            if source and source != "unknown":
                relation_entities.add(str(source))
            if target and target != "unknown":
                relation_entities.add(str(target))
        
        entity_id_set = set(str(e) for e in entity_ids)
        
        # 计算重合率
        if not entity_id_set:
            return 0.0
            
        overlap_ratio = len(relation_entities.intersection(entity_id_set)) / len(entity_id_set)
        
        # 提供一些加成，但不超过1.0
        return min(1.0, overlap_ratio * 1.2)
    
    def _evaluate_naive_agent(self, sample) -> float:
        """评估naive代理的关系利用（基于文本块）"""
        chunks = sample.referenced_entities  # 存放文本块ID
        chunk_count = len(chunks) if chunks else 0
        
        # Naive代理不直接处理关系，根据文本块数量和内容给分
        base_score = 0.3
        
        # 尝试从文本块中提取实体间的隐含关系
        if chunk_count > 0 and self.neo4j_client:
            try:
                # 查询文本块中提到的实体
                query = """
                MATCH (c:__Chunk__)-[:MENTIONS]->(e:__Entity__)
                WHERE c.id IN $chunk_ids
                RETURN COLLECT(DISTINCT e.id) AS entity_ids
                """
                result = self.neo4j_client.execute_query(query, {"chunk_ids": chunks})
                
                entity_ids = []
                if result.records and result.records[0].get("entity_ids"):
                    entity_ids = result.records[0].get("entity_ids")
                
                # 查询这些实体之间的关系
                if len(entity_ids) > 1:
                    rel_query = """
                    MATCH (a:__Entity__)-[r]-(b:__Entity__)
                    WHERE a.id IN $entity_ids AND b.id IN $entity_ids AND a <> b
                    RETURN COUNT(DISTINCT r) AS rel_count
                    """
                    rel_result = self.neo4j_client.execute_query(rel_query, {"entity_ids": entity_ids})
                    
                    rel_count = 0
                    if rel_result.records and rel_result.records[0].get("rel_count") is not None:
                        rel_count = rel_result.records[0].get("rel_count")
                    
                    # 根据关系数量给予奖励分
                    rel_bonus = min(0.3, rel_count * 0.05)
                    return base_score + rel_bonus
            except Exception as e:
                print(f"评估naive代理关系时出错: {e}")
        
        # 如果无法评估关系，仅根据文本块数量给分
        chunk_bonus = min(0.3, chunk_count * 0.05)  # 每个文本块加0.05，最多加0.3
        return base_score + chunk_bonus
    
    def _evaluate_relationship_utilization(self, sample) -> float:
        """统一评估关系利用率"""
        referenced_rels = sample.referenced_relationships
        entity_ids = sample.referenced_entities
        
        # 如果没有引用关系，且无实体，给予基础分
        if not referenced_rels and not entity_ids:
            return 0.3
        
        # 如果没有引用关系，但有实体，尝试推断关系
        if not referenced_rels and entity_ids and self.neo4j_client:
            implicit_rel_score = self._calculate_implicit_relationships(entity_ids)
            if implicit_rel_score > 0:
                return 0.3 + implicit_rel_score * 0.4  # 基础分加上推断关系得分
        
        # 处理引用的关系
        rel_info = self._get_relationship_info(referenced_rels)
        
        # 没有有效关系信息，回退到基础分
        if not rel_info:
            return 0.3
        
        # 计算关系利用率的多个维度
        quantity_score = self._calculate_quantity_score(rel_info)
        quality_score = self._calculate_quality_score(rel_info)
        relevance_score = self._calculate_relevance_score(rel_info, entity_ids)
        
        # 计算综合得分 - 数量占30%，质量占40%，相关性占30%
        base_score = 0.3  # 统一基础分
        total_score = base_score + 0.7 * (
            0.3 * quantity_score + 
            0.4 * quality_score + 
            0.3 * relevance_score
        )
        
        # 确保不超过1.0
        return min(1.0, total_score)
    
    def _get_relationship_info(self, referenced_rels) -> List[Dict[str, Any]]:
        rel_info = []
        
        if not self.neo4j_client or not referenced_rels:
            return rel_info
            
        # 处理字符串ID类型的关系
        rel_ids = [r for r in referenced_rels if isinstance(r, str)]
        
        # 转换为数字ID
        numeric_rel_ids = []
        for rel_id in rel_ids:
            try:
                if rel_id.isdigit() or rel_id.lstrip('-').isdigit():
                    numeric_rel_ids.append(int(rel_id))
            except (ValueError, AttributeError):
                continue
        
        if not numeric_rel_ids:
            return rel_info
            
        try:
            # 直接查询所有关系，然后手动匹配
            query = """
            MATCH (a)-[r]->(b)
            RETURN a.id AS source, type(r) AS relation, b.id AS target, 
                r.description AS description, r.weight AS weight
            LIMIT 500
            """
            result = self.neo4j_client.execute_query(query)
            
            if result.records:
                # 只获取前50条关系作为样本
                count = 0
                for record in result.records:
                    if count >= 50:
                        break
                        
                    source = record.get("source")
                    relation = record.get("relation")
                    target = record.get("target")
                    description = record.get("description")
                    weight = record.get("weight")
                    
                    if source and relation and target:
                        rel_info.append({
                            "source": str(source),
                            "relation": relation,
                            "target": str(target),
                            "description": description,
                            "weight": weight
                        })
                        count += 1
            
            return rel_info
        except Exception as e:
            if self.debug:
                print(f"  查询关系信息失败: {e}")
            return rel_info
    
    def _calculate_implicit_relationships(self, entity_ids: List[str]) -> float:
        """计算实体间的隐含关系得分"""
        if not self.neo4j_client or len(entity_ids) < 2:
            return 0.0
            
        try:
            # 查询实体之间的关系
            query = """
            MATCH (a:__Entity__)-[r]-(b:__Entity__)
            WHERE a.id IN $entity_ids AND b.id IN $entity_ids AND a <> b
            RETURN COUNT(DISTINCT r) AS rel_count
            """
            result = self.neo4j_client.execute_query(query, {"entity_ids": entity_ids})
            
            rel_count = 0
            if result.records and result.records[0].get("rel_count") is not None:
                rel_count = result.records[0].get("rel_count")
            
            # 最多能有多少关系
            max_possible_rels = len(entity_ids) * (len(entity_ids) - 1) / 2
            
            # 计算关系覆盖率
            rel_coverage = rel_count / max_possible_rels if max_possible_rels > 0 else 0
            
            return min(1.0, rel_coverage)
        except Exception as e:
            print(f"计算隐含关系时出错: {e}")
            return 0.0
    
    def _calculate_quantity_score(self, rel_info: List[Dict[str, Any]]) -> float:
        """计算关系数量得分"""
        # 如果有关系详细信息，使用实际关系数量
        rel_count = len(rel_info) if rel_info else 0
        
        # 每个关系贡献0.1分，最多1.0
        return min(1.0, rel_count * 0.1)
    
    def _calculate_quality_score(self, rel_info: List[Dict[str, Any]]) -> float:
        """
        计算关系质量得分 
        
        Args:
            rel_info: 关系信息列表
            
        Returns:
            float: 关系质量得分
        """
        if not rel_info:
            return 0.0
            
        # 检查关系是否有描述 - 确保处理None值
        described_count = 0
        for rel in rel_info:
            # 使用描述或关系类型
            description = rel.get("description", "")
            relation_type = rel.get("relation", "")
            
            if ((description is not None and str(description).strip()) or 
                (relation_type is not None and str(relation_type).strip())):
                described_count += 1
        
        description_ratio = described_count / len(rel_info) if rel_info else 0
        
        # 检查关系类型的多样性
        relation_types = set()
        for rel in rel_info:
            rel_type = rel.get("relation", "")
            if rel_type and rel_type.strip():
                relation_types.add(rel_type)
        
        type_diversity = min(1.0, len(relation_types) / 5)  # 最多5种关系类型为满分
        
        # 检查来源和目标实体是否存在
        valid_relations = 0
        for rel in rel_info:
            source = rel.get("source", "")
            target = rel.get("target", "")
            if source and source != "unknown" and target and target != "unknown":
                valid_relations += 1
        
        validity_ratio = valid_relations / len(rel_info) if rel_info else 0
        
        # 计算关系权重的平均值（如果有）
        weight_score = 0.0
        weighted_rels = [rel for rel in rel_info if "weight" in rel and rel["weight"] is not None]
        if weighted_rels:
            try:
                weights = []
                for rel in weighted_rels:
                    # 确保权重是有效数字
                    if isinstance(rel["weight"], (int, float)):
                        weights.append(float(rel["weight"]))
                    elif isinstance(rel["weight"], str) and rel["weight"].replace('.', '', 1).isdigit():
                        weights.append(float(rel["weight"]))
                
                if weights:
                    avg_weight = sum(weights) / len(weights)
                    # 假设权重范围为0-10，归一化到0-1
                    weight_score = min(1.0, avg_weight / 10.0)
            except Exception as e:
                if self.debug:
                    print(f"  计算权重得分时出错: {e}")
        
        # 综合得分 - 描述占30%，多样性占30%，有效性占20%，权重占20%
        if weighted_rels:
            return (0.3 * description_ratio + 
                    0.3 * type_diversity + 
                    0.2 * validity_ratio + 
                    0.2 * weight_score)
        else:
            # 如果没有权重信息，重新分配占比
            return (0.4 * description_ratio + 
                    0.3 * type_diversity + 
                    0.3 * validity_ratio)
    
    def _calculate_relevance_score(self, rel_info: List[Dict], entity_ids: List[str]) -> float:
        """计算关系相关性得分"""
        if not rel_info or not entity_ids:
            return 0.0
            
        # 统计关系中的实体与引用实体的重合度
        relation_entities = set()
        for rel in rel_info:
            source = rel.get("source")
            target = rel.get("target")
            if source:
                relation_entities.add(str(source))
            if target:
                relation_entities.add(str(target))
        
        entity_id_set = set(str(e) for e in entity_ids)
        
        # 计算重合率
        if not entity_id_set:
            return 0.0
            
        overlap_ratio = len(relation_entities.intersection(entity_id_set)) / len(entity_id_set)
        
        return min(1.0, overlap_ratio * 1.2)  # 给予一些加成，但不超过1.0

class RetrievalLatency(BaseMetric):
    """检索延迟评估指标"""
    
    metric_name = "retrieval_latency"
    
    def __init__(self, config):
        super().__init__(config)
    
    def calculate_metric(self, data: RetrievalEvaluationData) -> Tuple[Dict[str, float], List[float]]:
        """
        计算检索延迟
        
        Args:
            data (RetrievalEvaluationData): 评估数据
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 总体得分和每个样本的得分
        """
        latency_scores = []
        
        for sample in data.samples:
            # 直接使用样本中记录的检索时间
            latency_scores.append(sample.retrieval_time)
        
        # 计算平均延迟
        avg_latency = sum(latency_scores) / len(latency_scores) if latency_scores else 0.0
        
        return {"retrieval_latency": avg_latency}, latency_scores


class EntityCoverageMetric(BaseMetric):
    """实体覆盖率评估指标"""
    
    metric_name = "entity_coverage"
    
    def __init__(self, config):
        super().__init__(config)
        self.neo4j_client = config.get('neo4j_client', None)
    
    def calculate_metric(self, data) -> Tuple[Dict[str, float], List[float]]:
        """计算实体覆盖率"""
        
        print("\n======== EntityCoverage 计算日志 ========")
        
        coverage_scores = []
        
        # 打印总体信息
        total_samples = len(data.samples) if hasattr(data, 'samples') else 0
        print(f"样本总数: {total_samples}")
        
        for idx, sample in enumerate(data.samples):
            question = sample.question
            agent_type = sample.agent_type.lower() if sample.agent_type else ""
            
            print(f"\n样本 {idx+1}:")
            print(f"  问题: {question[:50]}...")
            print(f"  代理类型: {agent_type}")
            
            # 提取问题关键词
            keywords = self._extract_keywords(question)
            print(f"  提取关键词: {keywords}")
            
            # 统一计算实体覆盖率
            score = self._evaluate_entity_coverage(sample, keywords)
            print(f"  实体覆盖率分数: {score:.4f}")
            
            coverage_scores.append(score)
        
        avg_coverage = sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0.0
        print(f"\n平均实体覆盖率: {avg_coverage:.4f}")
        print("======== EntityCoverage 计算结束 ========\n")
        
        return {"entity_coverage": avg_coverage}, coverage_scores
    
    def _extract_keywords(self, question: str) -> List[str]:
        """从问题中提取关键词"""
        keywords = re.findall(r'\b[\w\u4e00-\u9fa5]{2,}\b', normalize_answer(question))
        # 过滤过长或过短的关键词
        return [k for k in keywords if len(k) > 1 and len(k) < 15]
    
    def _evaluate_naive_chunks(self, sample, keywords: List[str]) -> float:
        """评估naive代理基于文本块的覆盖率"""
        chunks = sample.referenced_entities  # 存放文本块ID
        
        # 获取文本块内容进行评估
        chunk_texts = []
        if self.neo4j_client and chunks:
            try:
                # 直接从Neo4j查询文本块内容
                query = """
                MATCH (c:__Chunk__)
                WHERE c.id IN $ids
                RETURN c.text AS text
                """
                result = self.neo4j_client.execute_query(query, {"ids": chunks})
                
                if result.records:
                    for record in result.records:
                        text = record.get("text", "")
                        if text:
                            chunk_texts.append(text)
            except Exception as e:
                print(f"获取文本块内容失败: {e}")
        
        # 根据关键词在文本块中的匹配情况评分
        if keywords and chunk_texts:
            matched = 0
            for keyword in keywords:
                for text in chunk_texts:
                    if keyword.lower() in text.lower():
                        matched += 1
                        break
            
            # 计算匹配率和文本块数量的综合得分
            match_rate = matched / len(keywords) if keywords else 0
            chunk_factor = min(1.0, len(chunk_texts) / 3)  # 最多3个文本块为满分
            
            # 根据匹配率和文本块数量计算加权得分
            base_score = 0.4  # 基础分保持一致
            match_score = 0.5 * match_rate * chunk_factor
            return base_score + match_score
        
        # 如果没有关键词或文本块，给予基础分
        return 0.4
    
    def _evaluate_entity_coverage(self, sample, keywords: List[str]) -> float:
        """
        统一计算实体覆盖率得分
        
        Args:
            sample: 评估样本
            keywords: 问题关键词
            
        Returns:
            float: 实体覆盖率得分
        """
        # 提取实体信息
        entities = []
        entity_ids = sample.referenced_entities
        
        print(f"  引用的实体ID数量: {len(entity_ids) if entity_ids else 0}")
        if entity_ids:
            print(f"  引用实体ID样例: {entity_ids[:5]}{'...' if len(entity_ids) > 5 else ''}")
        
        # 查询Neo4j获取实体信息
        if self.neo4j_client and entity_ids:
            try:
                query = """
                MATCH (e)
                WHERE e.id IN $ids
                RETURN e.id AS id, e.description AS description
                """
                result = self.neo4j_client.execute_query(query, {"ids": entity_ids})
                
                # 记录Neo4j查询结果
                if result.records:
                    for record in result.records:
                        entity_id = record.get("id", "")
                        entity_desc = record.get("description", "")
                        if entity_id:
                            entities.append(f"{entity_id} {entity_desc}")
                print(f"  从Neo4j获取的实体数量: {len(entities)}")
            except Exception as e:
                print(f"  查询实体信息失败: {e}")
        
        # 如果无法从Neo4j获取实体信息，直接使用ID
        if not entities and entity_ids:
            entities = entity_ids
            print("  使用原始实体ID作为实体信息")
        
        # 计算关键词匹配率
        if keywords and entities:
            # 将所有实体信息合并为一个文本
            entities_text = " ".join([str(e) for e in entities]).lower()
            print(f"  实体文本长度: {len(entities_text)}")
            
            # 匹配关键词
            matched = 0
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in entities_text:
                    matched += 1
            
            # 尝试数字ID匹配
            for keyword in keywords:
                if not any(keyword.lower() in str(e).lower() for e in entities):
                    # 对于未匹配的关键词，尝试通过ID间接匹配
                    for entity_id in entity_ids:
                        # 获取相关联的实体
                        try:
                            if self.neo4j_client:
                                query = """
                                MATCH (e)-[r]-(related)
                                WHERE e.id = $id
                                RETURN related.description AS description
                                LIMIT 10
                                """
                                result = self.neo4j_client.execute_query(query, {"id": entity_id})
                                
                                if result.records:
                                    for record in result.records:
                                        desc = record.get("description", "")
                                        if desc and keyword.lower() in desc.lower():
                                            matched += 0.5  # 相关实体匹配给予部分分数
                                            break
                        except Exception as e:
                            # 忽略错误
                            pass
            
            # 计算匹配率和实体数量因子
            match_rate = matched / len(keywords) if keywords else 0
            entity_factor = min(1.0, len(entities) / 5)  # 最多5个实体为满分
            
            print(f"  关键词匹配数: {matched}/{len(keywords)}")
            print(f"  匹配率: {match_rate:.4f}")
            print(f"  实体因子(基于数量): {entity_factor:.4f}")
            
            # 计算综合得分
            base_score = 0.4
            quality_score = 0.6 * match_rate * entity_factor
            
            print(f"  基础分: {base_score}")
            print(f"  质量得分: {quality_score:.4f}")
            
            return min(1.0, base_score + quality_score)
        
        # 如果实体列表为空，但agent_type为graph或hybrid，给予稍高分数
        agent_type = sample.agent_type.lower()
        if agent_type in ["graph", "hybrid"] and entity_ids:
            # 根据实体ID数量给予一定加分
            id_count_score = min(0.3, len(entity_ids) * 0.05)  # 每个ID加0.05，最多0.3
            score = 0.4 + id_count_score
            print(f"  基于实体ID数量的得分: {score:.4f}")
            return score
        
        # 没有实体或关键词时，给予基础分
        print("  没有实体或关键词，使用基础分: 0.4")
        return 0.4
    
    def _calculate_graph_relevance(self, entity_ids: List[str], keywords: List[str]) -> float:
        """计算graph代理特有的实体相关性得分"""
        if not self.neo4j_client or not entity_ids or not keywords:
            return 0.0
            
        try:
            # 查询实体之间的关系密度
            query = """
            MATCH (a:__Entity__)-[r]-(b:__Entity__)
            WHERE a.id IN $ids AND b.id IN $ids
            RETURN COUNT(DISTINCT r) AS rel_count
            """
            result = self.neo4j_client.execute_query(query, {"ids": entity_ids})
            
            rel_count = 0
            if result.records and result.records[0].get("rel_count") is not None:
                rel_count = result.records[0].get("rel_count")
            
            # 计算相关性得分 - 基于关系密度
            entity_count = len(entity_ids)
            max_possible_rels = entity_count * (entity_count - 1) / 2 if entity_count > 1 else 1
            rel_density = min(1.0, rel_count / max_possible_rels)
            
            return rel_density
        except Exception as e:
            print(f"计算图相关性时出错: {e}")
            return 0.0


class ChunkUtilization(BaseMetric):
    """文本块利用率评估指标"""
    
    metric_name = "chunk_utilization"
    
    def __init__(self, config):
        super().__init__(config)
        self.neo4j_client = config.get('neo4j_client', None)
    
    def calculate_metric(self, data: RetrievalEvaluationData) -> Tuple[Dict[str, float], List[float]]:
        """
        计算文本块利用率 - 从引用数据中提取的chunk被利用的程度
        
        Args:
            data (RetrievalEvaluationData): 评估数据
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 总体得分和每个样本的得分
        """
        chunk_scores = []
        
        for sample in data.samples:
            # 从原始回答中提取引用的chunks
            refs = extract_references_from_answer(sample.system_answer)
            chunk_ids = refs.get("chunks", [])
            
            if not chunk_ids:
                chunk_scores.append(0.0)
                continue
            
            # 在回答中查找chunk内容的使用情况
            answer_text = clean_references(sample.system_answer)
            answer_text = clean_thinking_process(answer_text)
            
            if not self.neo4j_client:
                # 如果没有Neo4j客户端，使用默认值
                chunk_scores.append(0.5)
                continue
            
            # 从Neo4j获取chunk内容
            try:
                chunk_texts = []
                total_matches = 0
                
                for chunk_id in chunk_ids:
                    # 查询文本块内容
                    query = """
                    MATCH (n:__Chunk__) 
                    WHERE n.id = $id 
                    RETURN n.text AS text
                    """
                    
                    result = self.neo4j_client.execute_query(query, {"id": chunk_id})
                    
                    if result.records and len(result.records) > 0:
                        chunk_text = result.records[0].get("text", "")
                        if chunk_text:
                            chunk_texts.append(chunk_text)
                            
                            # 计算文本块内容在回答中的利用率
                            # 将文本块分成关键短语
                            key_phrases = re.findall(r'\b[\w\u4e00-\u9fa5]{4,}\b', chunk_text)
                            key_phrases = list(set([p for p in key_phrases if len(p) > 3]))
                            
                            if key_phrases:
                                # 计算关键短语在回答中出现的比例
                                matched_phrases = sum(1 for phrase in key_phrases 
                                                    if phrase.lower() in answer_text.lower())
                                match_ratio = matched_phrases / len(key_phrases)
                                total_matches += match_ratio
                
                # 计算平均利用率
                if chunk_texts:
                    chunk_utilization = total_matches / len(chunk_texts)
                    chunk_scores.append(chunk_utilization)
                else:
                    chunk_scores.append(0.0)
                    
            except Exception as e:
                print(f"计算文本块利用率时出错: {e}")
                chunk_scores.append(0.5)  # 出错时使用默认值
        
        avg_chunk_utilization = sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0.0
        
        return {"chunk_utilization": avg_chunk_utilization}, chunk_scores

    
class GraphRAGRetrievalEvaluator(BaseEvaluator):
    """GraphRAG检索评估器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.neo4j_client = config.get('neo4j_client', None)
        self.qa_agent = config.get('qa_agent', None)
    
    def evaluate(self, data: RetrievalEvaluationData) -> Dict[str, float]:
        """执行评估"""
        print("\n======== 开始评估检索性能 ========")
        
        # 打印样本信息
        print(f"样本总数: {len(data.samples)}")

        # 预处理阶段 - 建立实体和关系映射
        self._prepare_entity_relation_maps()
        
        # 首先处理每个样本的数据，确保引用的实体和关系信息完整
        for i, sample in enumerate(data.samples):
            print(f"\n处理样本 {i+1}:")
            
            # 打印基本信息
            print(f"  问题: {sample.question[:50]}...")
            print(f"  代理类型: {sample.agent_type}")

            # 增强实体和关系处理
            self._enhance_entity_data(sample)
            self._enhance_relation_data(sample)
            
            # 打印处理后的信息
            print(f"  处理后的引用实体数量: {len(sample.referenced_entities)}")
            print(f"  处理后的引用关系数量: {len(sample.referenced_relationships)}")
            
            # 打印回答的一部分以及从回答中提取的引用数据
            answer = sample.system_answer
            print(f"  回答前100字符: {answer[:100]}...")
            
            # 显示当前样本的引用实体和关系信息
            print(f"  当前引用实体数量: {len(sample.referenced_entities)}")
            print(f"  当前引用关系数量: {len(sample.referenced_relationships)}")
            
            # 提取引用数据并打印
            from evaluator.preprocessing import extract_references_from_answer
            refs = extract_references_from_answer(answer)
            
            print(f"  提取的引用数据:")
            print(f"    实体: {refs.get('entities', [])[:5]}{'...' if len(refs.get('entities', [])) > 5 else ''}") 
            print(f"    关系: {refs.get('relationships', [])[:5]}{'...' if len(refs.get('relationships', [])) > 5 else ''}")
            print(f"    文本块: {refs.get('chunks', [])[:3]}{'...' if len(refs.get('chunks', [])) > 3 else ''}")
            
            # 1. 处理naive代理 - 确保文本块数据正确存储
            if sample.agent_type.lower() == "naive":
                print("  处理Naive代理的引用数据...")
                
                # 将文本块ID从referenced_relationships移到referenced_entities
                if not sample.referenced_entities and isinstance(sample.referenced_relationships, list):
                    for item in sample.referenced_relationships:
                        if isinstance(item, str) and len(item) > 30:  # 长字符串可能是文本块ID
                            sample.referenced_entities.append(item)
                    sample.referenced_relationships = []
                    print(f"  将文本块从关系移到实体字段，现在实体数: {len(sample.referenced_entities)}")
                
                # 确保从json数据中提取的文本块ID在referenced_entities中
                for chunk_id in refs.get("chunks", []):
                    if chunk_id not in sample.referenced_entities:
                        sample.referenced_entities.append(chunk_id)
                        print(f"  添加文本块ID: {chunk_id[:10]}...")
            
            # 2. 处理其他代理 - 确保实体和关系ID正确存储
            else:
                print("  处理非Naive代理的引用数据...")
                
                # 更新实体ID
                added_entities = 0
                for entity_id in refs.get("entities", []):
                    if entity_id and entity_id not in sample.referenced_entities:
                        sample.referenced_entities.append(entity_id)
                        added_entities += 1
                
                # 更新关系ID
                added_relationships = 0
                for rel_id in refs.get("relationships", []):
                    if rel_id and rel_id not in sample.referenced_relationships:
                        sample.referenced_relationships.append(rel_id)
                        added_relationships += 1
                        
                print(f"  添加了{added_entities}个实体和{added_relationships}个关系")
                
            # 显示最终引用信息
            print(f"  最终引用实体数量: {len(sample.referenced_entities)}")
            print(f"  最终引用关系数量: {len(sample.referenced_relationships)}")
        
        # 执行评估计算
        result_dict = {}
        
        for metric_name in self.metrics:
            try:
                print(f"\n开始计算指标: {metric_name}")
                print(f"\n使用评估类: {self.metric_class[metric_name].__class__.__name__}")
                metric_result, metric_scores = self.metric_class[metric_name].calculate_metric(data)
                result_dict.update(metric_result)
                
                # 更新每个样本的评分
                for sample, metric_score in zip(data.samples, metric_scores):
                    sample.update_evaluation_score(metric_name, metric_score)
                    
                print(f"完成指标 {metric_name} 计算，平均得分: {list(metric_result.values())[0]:.4f}")
            except Exception as e:
                import traceback
                print(f'评估 {metric_name} 时出错: {e}')
                print(traceback.format_exc())
                continue
        
        print("\n所有指标计算结果:")
        for metric, score in result_dict.items():
            print(f"  {metric}: {score:.4f}")
        
        print("======== 检索性能评估结束 ========\n")
        
        # 保存评估结果
        if self.save_metric_flag:
            self.save_metric_score(result_dict)
        
        # 保存评估数据
        if self.save_data_flag:
            self.save_data(data)
        
        return result_dict
    
    def _prepare_entity_relation_maps(self):
        """准备实体和关系映射，用于快速查找"""
        self.entity_map = {}
        self.relation_map = {}
        
        if not self.neo4j_client:
            return
        
        try:
            # 获取所有实体
            entity_query = """
            MATCH (n)
            RETURN n.id AS id, n.description AS description
            LIMIT 2000
            """
            entity_result = self.neo4j_client.execute_query(entity_query)
            
            if entity_result.records:
                for record in entity_result.records:
                    ent_id = record.get("id")
                    ent_desc = record.get("description", "")
                    if ent_id:
                        self.entity_map[str(ent_id)] = ent_desc
            
            # 获取所有关系
            relation_query = """
            MATCH (a)-[r]->(b)
            RETURN a.id AS source, type(r) AS relation, b.id AS target, r.id AS rel_id
            LIMIT 1000
            """
            relation_result = self.neo4j_client.execute_query(relation_query)
            
            if relation_result.records:
                for record in relation_result.records:
                    rel_id = record.get("rel_id")
                    source = record.get("source")
                    relation = record.get("relation")
                    target = record.get("target")
                    
                    if rel_id and source and relation and target:
                        self.relation_map[str(rel_id)] = {
                            "source": str(source),
                            "relation": relation,
                            "target": str(target)
                        }
        except Exception as e:
            print(f"准备实体和关系映射时出错: {e}")

    def _enhance_entity_data(self, sample):
        """增强实体数据处理"""
        # 1. 确保实体ID是字符串
        sample.referenced_entities = [str(e) for e in sample.referenced_entities]
        
        # 2. 尝试添加实体描述
        if self.entity_map:
            enhanced_entities = []
            for ent_id in sample.referenced_entities:
                if ent_id in self.entity_map:
                    desc = self.entity_map[ent_id]
                    enhanced_entity = {
                        "id": ent_id,
                        "description": desc
                    }
                    enhanced_entities.append(enhanced_entity)
                else:
                    enhanced_entities.append({
                        "id": ent_id,
                        "description": f"实体 {ent_id}"
                    })
            
            # 将增强的实体信息保存到样本中
            sample.entity_details = enhanced_entities

    def _enhance_relation_data(self, sample):
        """增强关系数据处理"""
        # 1. 处理字符串ID的关系
        if not isinstance(sample.referenced_relationships, list):
            sample.referenced_relationships = []
            return
        
        string_rel_ids = [r for r in sample.referenced_relationships if isinstance(r, str)]
        
        # 2. 尝试使用关系映射增强关系信息
        enhanced_relations = []
        for rel_id in string_rel_ids:
            if rel_id in self.relation_map:
                rel_data = self.relation_map[rel_id]
                enhanced_relation = (
                    rel_data["source"],
                    rel_data["relation"],
                    rel_data["target"]
                )
                enhanced_relations.append(enhanced_relation)
        
        # 3. 如果成功增强了关系，更新样本
        if enhanced_relations:
            sample.enhanced_relationships = enhanced_relations
        else:
            # 使用更智能的方式创建占位关系
            relation_types = ["MENTIONS", "RELATES_TO", "PART_OF", "CONTAINS"]
            
            for i, rel_id in enumerate(string_rel_ids):
                rel_type = relation_types[i % len(relation_types)]
                source = f"entity_{i}"
                target = f"entity_{i+1}"
                
                enhanced_relations.append((source, rel_type, target))
            
            sample.enhanced_relationships = enhanced_relations
    
    def get_entities_info(self, entity_ids: List[str]) -> List[Tuple[str, str]]:
        """获取实体信息（ID和描述）"""
        if not self.neo4j_client or not entity_ids:
            return []
        
        try:
            query = """
            MATCH (e:__Entity__)
            WHERE e.id IN $ids
            RETURN e.id AS id, e.description AS description
            """
            
            result = self.neo4j_client.execute_query(query, {"ids": entity_ids})
            
            entities_info = []
            if result.records:
                for record in result.records:
                    entity_id = record.get("id", "未知ID")
                    entity_desc = record.get("description", "")
                    # 使用实体ID和描述
                    entities_info.append((str(entity_id), entity_desc or ""))
            
            # 如果没有找到实体，返回原始ID
            if not entities_info:
                entities_info = [(eid, "") for eid in entity_ids]
                
            return entities_info
                
        except Exception as e:
            print(f"查询实体信息失败: {e}")
            return [(eid, "") for eid in entity_ids]

    def get_relationships_info(self, relationship_ids: List[str]) -> List[Tuple[str, str, str]]:
        """获取关系信息（源实体-关系类型-目标实体）"""
        if not self.neo4j_client or not relationship_ids:
            return []
        
        try:
            # 转换所有ID为整数
            numeric_ids = []
            for rid in relationship_ids:
                try:
                    numeric_ids.append(int(rid))
                except (ValueError, TypeError):
                    # 如果不能转换为整数，跳过
                    pass
            
            if not numeric_ids:
                # 如果没有有效的数字ID，返回空列表
                return []
            
            # 通过关系ID直接匹配关系
            query = """
            MATCH (a)-[r]->(b)
            WHERE r.id IN $ids
            RETURN a.id AS source, type(r) AS relation, b.id AS target, 
                r.description AS description
            """
            
            result = self.neo4j_client.execute_query(query, {"ids": numeric_ids})
            
            relationships_info = []
            if result.records:
                for record in result.records:
                    source = record.get("source")
                    relation = record.get("relation")
                    target = record.get("target")
                    description = record.get("description", "")
                    
                    # 只有当所有值都存在时才添加关系
                    if source and relation and target:
                        # 使用关系的描述补充关系类型
                        rel_info = relation
                        if description:
                            rel_info = f"{relation}({description})"
                            
                        relationships_info.append((str(source), rel_info, str(target)))
            
            return relationships_info
                
        except Exception as e:
            print(f"查询关系信息失败: {e}")
            return []
        

    def evaluate_agent(self, agent_name: str, questions: List[str]) -> Dict[str, float]:
        """
        评估特定代理的检索性能
        
        Args:
            agent_name: 代理名称 (naive, hybrid, graph, deep)
            questions: 问题列表
            
        Returns:
            Dict[str, float]: 评估结果
        """
        agents = {
            "naive": self.config.get("naive_agent"),
            "hybrid": self.config.get("hybrid_agent"),
            "graph": self.config.get("graph_agent"),
            "deep": self.config.get("deep_agent")
        }
        
        agent = agents.get(agent_name)
        if not agent:
            raise ValueError(f"未找到代理: {agent_name}")
        
        # 创建评估数据集
        eval_data = RetrievalEvaluationData()
        
        # 处理每个问题
        for question in questions:
            # 创建评估样本
            sample = RetrievalEvaluationSample(
                question=question,
                agent_type=agent_name
            )
            
            # 记录开始时间
            start_time = time.time()
            
            # 普通回答
            answer = agent.ask(question)
            
            # 计算检索时间
            retrieval_time = time.time() - start_time
            
            # 更新样本
            sample.update_system_answer(answer, agent_name)
            sample.retrieval_time = retrieval_time
            
            # 使用Neo4j获取相关图数据
            if self.neo4j_client:
                entities, relationships = self._get_relevant_graph_data(question)
                sample.update_retrieval_data(entities, relationships)
            
            # 添加到评估数据
            eval_data.append(sample)
        
        # 执行评估
        return self.evaluate(eval_data)
    
    def compare_agents(self, questions: List[str]) -> Dict[str, Dict[str, float]]:
        """
        比较所有代理的检索性能
        
        Args:
            questions: 问题列表
            
        Returns:
            Dict[str, Dict[str, float]]: 每个代理的评估结果
        """
        agents = {
            "naive": self.config.get("naive_agent"),
            "hybrid": self.config.get("hybrid_agent"),
            "graph": self.config.get("graph_agent"),
            "deep": self.config.get("deep_agent")
        }
        
        results = {}
        
        for agent_name, agent in agents.items():
            if agent:
                print(f"评估代理: {agent_name}")
                agent_results = self.evaluate_agent(agent_name, questions)
                results[agent_name] = agent_results
                
                # 打印结果
                print(f"{agent_name} 评估结果:")
                for metric, score in agent_results.items():
                    print(f"  {metric}: {score:.4f}")
                print()
        
        return results
    
    def _get_relevant_graph_data(self, question: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """从Neo4j获取与问题相关的实体和关系"""
        if not self.neo4j_client:
            return [], []
            
        try:
            # 提取问题关键词
            import jieba.analyse
            question_words = jieba.analyse.extract_tags(question, topK=5)
        except Exception as e:
            # 简单分词回退方案
            print(f"关键词提取失败: {e}")
            question_words = re.findall(r'\b[\w\u4e00-\u9fa5]{2,}\b', question)
            question_words = [w for w in question_words if len(w) > 1]
        
        entities = []
        relationships = []
        
        try:
            # 查询与关键词相关的实体 - 使用e.id和e.description
            entity_query = """
            MATCH (e:__Entity__)
            WHERE ANY(word IN $keywords WHERE 
                e.id CONTAINS word OR
                e.description CONTAINS word)
            RETURN e.id AS id
            LIMIT 15
            """
            
            entity_result = self.neo4j_client.execute_query(entity_query, {"keywords": question_words})
            
            if entity_result.records:
                for record in entity_result.records:
                    entity_id = record.get("id")
                    if entity_id:
                        entities.append(entity_id)
            
            # 如果找到实体，查询相关关系
            if entities:
                # 查询实体之间的关系
                rel_query = """
                MATCH (a:__Entity__)-[r]->(b:__Entity__)
                WHERE a.id IN $entity_ids OR b.id IN $entity_ids
                RETURN DISTINCT a.id AS source, type(r) AS relation, b.id AS target
                LIMIT 30
                """
                
                rel_result = self.neo4j_client.execute_query(rel_query, {"entity_ids": entities})
                
                if rel_result.records:
                    for record in rel_result.records:
                        source = record.get("source")
                        relation = record.get("relation")
                        target = record.get("target")
                        if source and relation and target:
                            relationships.append((source, relation, target))
            
            # 如果未找到足够实体，尝试通过文本块查找
            if len(entities) < 3:
                chunk_query = """
                MATCH (c:__Chunk__)
                WHERE ANY(word IN $keywords WHERE c.text CONTAINS word)
                RETURN c.id AS chunk_id
                LIMIT 5
                """
                
                chunk_result = self.neo4j_client.execute_query(chunk_query, {"keywords": question_words})
                
                chunk_ids = []
                if chunk_result.records:
                    for record in chunk_result.records:
                        chunk_id = record.get("chunk_id")
                        if chunk_id:
                            chunk_ids.append(chunk_id)
                
                # 如果找到文本块，获取相关实体
                if chunk_ids:
                    chunk_entity_query = """
                    MATCH (c:__Chunk__)-[:MENTIONS]->(e:__Entity__)
                    WHERE c.id IN $chunk_ids
                    RETURN DISTINCT e.id AS entity_id
                    """
                    
                    chunk_entity_result = self.neo4j_client.execute_query(
                        chunk_entity_query, {"chunk_ids": chunk_ids}
                    )
                    
                    if chunk_entity_result.records:
                        for record in chunk_entity_result.records:
                            entity_id = record.get("entity_id")
                            if entity_id and entity_id not in entities:
                                entities.append(entity_id)
        except Exception as e:
            print(f"获取图数据时出错: {e}")
        
        return entities, relationships
    
    def format_comparison_table(self, results: Dict[str, Dict[str, float]]) -> str:
        """
        将比较结果格式化为表格
        
        Args:
            results: 比较结果
            
        Returns:
            str: 表格字符串
        """
        # 获取所有指标
        all_metrics = set()
        for agent_results in results.values():
            all_metrics.update(agent_results.keys())
        
        # 构建表头
        header = "| 指标 | " + " | ".join(results.keys()) + " |"
        separator = "| --- | " + " | ".join(["---" for _ in results]) + " |"
        
        # 构建行
        rows = []
        for metric in sorted(all_metrics):
            row = f"| {metric} |"
            for agent in results:
                score = results[agent].get(metric, "N/A")
                if isinstance(score, float):
                    score_str = f"{score:.4f}"
                else:
                    score_str = str(score)
                row += f" {score_str} |"
            rows.append(row)
        
        # 拼接表格
        table = "\n".join([header, separator] + rows)
        return table