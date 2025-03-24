import re
from collections import Counter
from typing import Dict, List, Union, Tuple, Any, Optional

from evaluator.utils import normalize_answer

class BaseMetric:
    """
    所有评估指标的基类，定义了评估指标的基本接口
    """
    metric_name = "base"

    def __init__(self, config: Dict):
        self.config = config
        self.dataset_name = config.get('dataset_name', 'default')

    def calculate_metric(self, data: Any) -> Tuple[Dict, List]:
        """
        计算评估指标和每个数据样本的得分
        
        Args:
            data: 包含基本信息和生成信息的数据对象
            
        Returns:
            (metric_score, metric_score_list): 
            - metric_score: 评估指标总分，例如 {"f1": 0.75}
            - metric_score_list: 每个样本的得分列表
        """
        return {}, []


class F1_Score(BaseMetric):
    """
    计算预测和真实答案之间的F1分数，这是精确度和召回率的调和平均数
    """
    metric_name = "f1"

    def __init__(self, config: Dict):
        super().__init__(config)

    def token_level_scores(self, prediction: str, ground_truths: Union[str, List[str]]) -> Dict[str, float]:
        """
        计算单个预测与一个或多个真实答案之间的token级F1分数
        
        Args:
            prediction: 预测的文本
            ground_truths: 真实的答案，可以是单个字符串或字符串列表
            
        Returns:
            包含 'f1', 'precision', 和 'recall' 的字典
        """
        final_metric = {'f1': 0, 'precision': 0, 'recall': 0}
        
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]
            
        for ground_truth in ground_truths:
            normalized_prediction = normalize_answer(prediction)
            normalized_ground_truth = normalize_answer(ground_truth)

            # 如果预测或真实答案为特定单词（是，否，无答案），且不匹配，则跳过
            if normalized_prediction in ['是', '否', '不知道', 'yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            if normalized_ground_truth in ['是', '否', '不知道', 'yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue

            # 将预测的答案和真实的答案分割成单词（tokens）
            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()

            # 计算两组tokens的共同部分
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            
            if num_same == 0:
                continue

            # 计算精确度
            precision = 1.0 * num_same / len(prediction_tokens)
            
            # 计算召回率
            recall = 1.0 * num_same / len(ground_truth_tokens)
            
            # 计算F1分数
            f1 = (2 * precision * recall) / (precision + recall)
            
            # 更新最终的指标
            for k in ['f1', 'precision', 'recall']:
                final_metric[k] = max(eval(k), final_metric[k])
                
        return final_metric

    def calculate_metric(self, data: Any) -> Tuple[Dict[str, float], List[float]]:
        """
        计算数据集上的F1分数
        
        Args:
            data: 包含预测答案和真实答案的数据集
            
        Returns:
            tuple: 包含总体F1分数和每个样本的F1分数列表
        """
        pred_list = data.pred
        golden_answers_list = data.golden_answers
        
        metric_score_list = [
            self.token_level_scores(pred, golden_answers)['f1'] 
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        
        f1 = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0
        
        return {"f1": f1}, metric_score_list


class Recall_Score(F1_Score):
    """
    基于分词的召回率得分，继承自F1_Score类
    """
    metric_name = "recall"

    def __init__(self, config: Dict):
        super().__init__(config)

    def calculate_metric(self, data: Any) -> Tuple[Dict[str, float], List[float]]:
        """
        计算数据集上的召回率得分
        
        Args:
            data: 包含预测答案和真实答案的数据集
            
        Returns:
            tuple: 包含总体召回率得分和每个样本的召回率得分列表
        """
        # 从数据集中获取所有预测答案
        pred_list = data.pred
        
        # 从数据集中获取所有真实答案
        golden_answers_list = data.golden_answers
        
        # 计算每个样本的召回率
        metric_score_list = [
            self.token_level_scores(pred, golden_answers)['recall'] 
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]

        # 计算平均召回率
        recall = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0

        return {"recall": recall}, metric_score_list


class Precision_Score(F1_Score):
    """
    基于分词的精确度得分，继承自F1_Score类
    """
    metric_name = "precision"

    def __init__(self, config: Dict):
        super().__init__(config)

    def calculate_metric(self, data: Any) -> Tuple[Dict[str, float], List[float]]:
        """
        计算数据集上的精确度得分
        
        Args:
            data: 包含预测答案和真实答案的数据集
            
        Returns:
            tuple: 包含总体精确度得分和每个样本的精确度得分列表
        """
        # 从数据集中获取所有预测答案
        pred_list = data.pred
        
        # 从数据集中获取所有真实答案
        golden_answers_list = data.golden_answers
        
        # 计算每个样本的精确度
        metric_score_list = [
            self.token_level_scores(pred, golden_answers)['precision'] 
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]

        # 计算平均精确度
        precision = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0

        return {"precision": precision}, metric_score_list


class ExactMatch(BaseMetric):
    """
    精确匹配（ExactMatch, EM）指标，测量预测答案与标准答案是否完全一致
    """
    metric_name = "em"

    def __init__(self, config: Dict):
        super().__init__(config)
        self.is_regex = self.dataset_name == 'curatedtrec'

    def calculate_em(self, prediction: str, golden_answers: Union[str, List[str]]) -> float:
        """
        计算单个预测的精确匹配得分
        
        Args:
            prediction: 模型生成的预测文本
            golden_answers: 可能的正确答案列表或单个答案
            
        Returns:
            预测的精确匹配得分，1.0 表示完全匹配，0.0 表示不匹配
        """
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
            
        normalized_prediction = normalize_answer(prediction)
        score = 0.0
        
        for golden_answer in golden_answers:
            # 如果答案应视为正则表达式，则以正则表达式方式匹配
            if self.is_regex:
                golden_answer = re.compile(golden_answer, re.IGNORECASE)
                match = re.fullmatch(golden_answer, normalized_prediction)
                if match is not None:
                    score = 1.0
                    break
            else:
                golden_answer = normalize_answer(golden_answer)
                if golden_answer == normalized_prediction:
                    score = 1.0
                    break
                    
        return score

    def calculate_metric(self, data: Any) -> Tuple[Dict[str, float], List[float]]:
        """
        计算数据集上的精确匹配分数
        
        Args:
            data: 包含预测答案和真实答案的数据集
            
        Returns:
            tuple: 包含总体精确匹配分数和每个样本的精确匹配分数列表
        """
        golden_answers_list = data.golden_answers
        pred_list = data.pred

        metric_score_list = [
            self.calculate_em(pred, golden_answers) 
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        
        em_score = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0

        return {"em": em_score}, metric_score_list


class Sub_ExactMatch(BaseMetric):
    """
    子串精确匹配，检查预测答案是否包含标准答案
    """
    metric_name = "sub_em"

    def __init__(self, config: Dict):
        super().__init__(config)
        self.is_regex = self.dataset_name == 'curatedtrec'

    def calculate_sub_em(self, prediction: str, golden_answers: Union[str, List[str]]) -> float:
        """
        计算单个预测的子串精确匹配得分
        
        Args:
            prediction: 模型生成的预测文本
            golden_answers: 可能的正确答案列表或单个答案
            
        Returns:
            预测的子串精确匹配得分，1.0 表示包含匹配，0.0 表示不包含
        """
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
            
        normalized_prediction = normalize_answer(prediction)
        score = 0.0
        
        for golden_answer in golden_answers:
            if self.is_regex:
                golden_answer = re.compile(golden_answer, re.IGNORECASE)
                match = re.search(golden_answer, normalized_prediction)
                if match is not None:
                    score = 1.0
                    break
            else:
                golden_answer = normalize_answer(golden_answer)
                if golden_answer in normalized_prediction:
                    score = 1.0
                    break
                    
        return score

    def calculate_metric(self, data: Any) -> Tuple[Dict[str, float], List[float]]:
        """
        计算数据集上的子串精确匹配分数
        
        Args:
            data: 包含预测答案和真实答案的数据集
            
        Returns:
            tuple: 包含总体子串精确匹配分数和每个样本的子串精确匹配分数列表
        """
        golden_answers_list = data.golden_answers
        pred_list = data.pred

        metric_score_list = [
            self.calculate_sub_em(pred, golden_answers) 
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        
        sub_em_score = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0

        return {"sub_em": sub_em_score}, metric_score_list


class EntityRecall(BaseMetric):
    """
    实体召回率，评估生成的回答中包含了多少真实实体
    """
    metric_name = "entity_recall"

    def __init__(self, config: Dict):
        super().__init__(config)

    def calculate_metric(self, data: Any) -> Tuple[Dict[str, float], List[float]]:
        """
        计算实体召回率
        
        Args:
            data: 包含预测答案和真实答案中实体的数据集
            
        Returns:
            tuple: 包含总体实体召回率和每个样本的实体召回率列表
        """
        # 确保数据对象有所需的属性
        if not hasattr(data, 'pred_entities') or not hasattr(data, 'golden_entities'):
            return {"entity_recall": 0.0}, [0.0] * (len(data.pred) if hasattr(data, 'pred') else 0)
        
        pred_entities_list = data.pred_entities
        golden_entities_list = data.golden_entities
        
        metric_score_list = []
        
        for pred_entities, golden_entities in zip(pred_entities_list, golden_entities_list):
            if not golden_entities:  # 避免除以零
                metric_score_list.append(1.0)  # 如果没有黄金实体，则认为召回率为100%
                continue
                
            # 计算正确召回的实体数量
            recalled_entities = set(pred_entities) & set(golden_entities)
            recall_score = len(recalled_entities) / len(golden_entities)
            metric_score_list.append(recall_score)
        
        entity_recall = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0

        return {"entity_recall": entity_recall}, metric_score_list


class EntityPrecision(BaseMetric):
    """
    实体精确度，评估生成的回答中有多少实体是正确的
    """
    metric_name = "entity_precision"

    def __init__(self, config: Dict):
        super().__init__(config)

    def calculate_metric(self, data: Any) -> Tuple[Dict[str, float], List[float]]:
        """
        计算实体精确度
        
        Args:
            data: 包含预测答案和真实答案中实体的数据集
            
        Returns:
            tuple: 包含总体实体精确度和每个样本的实体精确度列表
        """
        # 确保数据对象有所需的属性
        if not hasattr(data, 'pred_entities') or not hasattr(data, 'golden_entities'):
            return {"entity_precision": 0.0}, [0.0] * (len(data.pred) if hasattr(data, 'pred') else 0)
        
        pred_entities_list = data.pred_entities
        golden_entities_list = data.golden_entities
        
        metric_score_list = []
        
        for pred_entities, golden_entities in zip(pred_entities_list, golden_entities_list):
            if not pred_entities:  # 避免除以零
                metric_score_list.append(0.0)  # 如果没有预测实体，则认为精确度为0
                continue
                
            # 计算正确预测的实体数量
            correct_entities = set(pred_entities) & set(golden_entities)
            precision_score = len(correct_entities) / len(pred_entities)
            metric_score_list.append(precision_score)
        
        entity_precision = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0

        return {"entity_precision": entity_precision}, metric_score_list


class EntityF1(BaseMetric):
    """
    实体F1分数，实体精确度和召回率的调和平均
    """
    metric_name = "entity_f1"

    def __init__(self, config: Dict):
        super().__init__(config)
        self.precision_metric = EntityPrecision(config)
        self.recall_metric = EntityRecall(config)

    def calculate_metric(self, data: Any) -> Tuple[Dict[str, float], List[float]]:
        """
        计算实体F1分数
        
        Args:
            data: 包含预测答案和真实答案中实体的数据集
            
        Returns:
            tuple: 包含总体实体F1分数和每个样本的实体F1分数列表
        """
        # 计算精确度和召回率
        precision_result, precision_scores = self.precision_metric.calculate_metric(data)
        recall_result, recall_scores = self.recall_metric.calculate_metric(data)
        
        entity_precision = precision_result["entity_precision"]
        entity_recall = recall_result["entity_recall"]
        
        # 计算样本级别的F1分数
        f1_scores = []
        for p, r in zip(precision_scores, recall_scores):
            if p + r == 0:  # 避免除以零
                f1_scores.append(0.0)
            else:
                f1_scores.append(2 * p * r / (p + r))
        
        # 计算总体F1分数
        if entity_precision + entity_recall == 0:
            entity_f1 = 0.0
        else:
            entity_f1 = 2 * entity_precision * entity_recall / (entity_precision + entity_recall)

        return {"entity_f1": entity_f1}, f1_scores


class RelationshipRecall(BaseMetric):
    """
    关系召回率，评估生成的回答中包含了多少真实关系
    """
    metric_name = "relationship_recall"

    def __init__(self, config: Dict):
        super().__init__(config)

    def calculate_metric(self, data: Any) -> Tuple[Dict[str, float], List[float]]:
        """
        计算关系召回率
        
        Args:
            data: 包含预测答案和真实答案中关系的数据集
            
        Returns:
            tuple: 包含总体关系召回率和每个样本的关系召回率列表
        """
        # 确保数据对象有所需的属性
        if not hasattr(data, 'pred_relationships') or not hasattr(data, 'golden_relationships'):
            return {"relationship_recall": 0.0}, [0.0] * (len(data.pred) if hasattr(data, 'pred') else 0)
        
        pred_relationships_list = data.pred_relationships
        golden_relationships_list = data.golden_relationships
        
        metric_score_list = []
        
        for pred_relationships, golden_relationships in zip(pred_relationships_list, golden_relationships_list):
            if not golden_relationships:  # 避免除以零
                metric_score_list.append(1.0)  # 如果没有黄金关系，则认为召回率为100%
                continue
                
            # 计算正确召回的关系数量
            recalled_relationships = set(pred_relationships) & set(golden_relationships)
            recall_score = len(recalled_relationships) / len(golden_relationships)
            metric_score_list.append(recall_score)
        
        relationship_recall = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0

        return {"relationship_recall": relationship_recall}, metric_score_list


class RelationshipPrecision(BaseMetric):
    """
    关系精确度，评估生成的回答中有多少关系是正确的
    """
    metric_name = "relationship_precision"

    def __init__(self, config: Dict):
        super().__init__(config)

    def calculate_metric(self, data: Any) -> Tuple[Dict[str, float], List[float]]:
        """
        计算关系精确度
        
        Args:
            data: 包含预测答案和真实答案中关系的数据集
            
        Returns:
            tuple: 包含总体关系精确度和每个样本的关系精确度列表
        """
        # 确保数据对象有所需的属性
        if not hasattr(data, 'pred_relationships') or not hasattr(data, 'golden_relationships'):
            return {"relationship_precision": 0.0}, [0.0] * (len(data.pred) if hasattr(data, 'pred') else 0)
        
        pred_relationships_list = data.pred_relationships
        golden_relationships_list = data.golden_relationships
        
        metric_score_list = []
        
        for pred_relationships, golden_relationships in zip(pred_relationships_list, golden_relationships_list):
            if not pred_relationships:  # 避免除以零
                metric_score_list.append(0.0)  # 如果没有预测关系，则认为精确度为0
                continue
                
            # 计算正确预测的关系数量
            correct_relationships = set(pred_relationships) & set(golden_relationships)
            precision_score = len(correct_relationships) / len(pred_relationships)
            metric_score_list.append(precision_score)
        
        relationship_precision = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0

        return {"relationship_precision": relationship_precision}, metric_score_list


class RelationshipF1(BaseMetric):
    """
    关系F1分数，关系精确度和召回率的调和平均
    """
    metric_name = "relationship_f1"

    def __init__(self, config: Dict):
        super().__init__(config)
        self.precision_metric = RelationshipPrecision(config)
        self.recall_metric = RelationshipRecall(config)

    def calculate_metric(self, data: Any) -> Tuple[Dict[str, float], List[float]]:
        """
        计算关系F1分数
        
        Args:
            data: 包含预测答案和真实答案中关系的数据集
            
        Returns:
            tuple: 包含总体关系F1分数和每个样本的关系F1分数列表
        """
        # 计算精确度和召回率
        precision_result, precision_scores = self.precision_metric.calculate_metric(data)
        recall_result, recall_scores = self.recall_metric.calculate_metric(data)
        
        relationship_precision = precision_result["relationship_precision"]
        relationship_recall = recall_result["relationship_recall"]
        
        # 计算样本级别的F1分数
        f1_scores = []
        for p, r in zip(precision_scores, recall_scores):
            if p + r == 0:  # 避免除以零
                f1_scores.append(0.0)
            else:
                f1_scores.append(2 * p * r / (p + r))
        
        # 计算总体F1分数
        if relationship_precision + relationship_recall == 0:
            relationship_f1 = 0.0
        else:
            relationship_f1 = 2 * relationship_precision * relationship_recall / (relationship_precision + relationship_recall)

        return {"relationship_f1": relationship_f1}, f1_scores


class CommunityRecall(BaseMetric):
    """
    社区召回率，评估生成的回答中包含了多少真实社区
    """
    metric_name = "community_recall"

    def __init__(self, config: Dict):
        super().__init__(config)

    def calculate_metric(self, data: Any) -> Tuple[Dict[str, float], List[float]]:
        """
        计算社区召回率
        
        Args:
            data: 包含预测答案和真实答案中社区的数据集
            
        Returns:
            tuple: 包含总体社区召回率和每个样本的社区召回率列表
        """
        # 确保数据对象有所需的属性
        if not hasattr(data, 'pred_communities') or not hasattr(data, 'golden_communities'):
            return {"community_recall": 0.0}, [0.0] * (len(data.pred) if hasattr(data, 'pred') else 0)
        
        pred_communities_list = data.pred_communities
        golden_communities_list = data.golden_communities
        
        metric_score_list = []
        
        for pred_communities, golden_communities in zip(pred_communities_list, golden_communities_list):
            if not golden_communities:  # 避免除以零
                metric_score_list.append(1.0)  # 如果没有黄金社区，则认为召回率为100%
                continue
                
            # 计算正确召回的社区数量
            recalled_communities = set(pred_communities) & set(golden_communities)
            recall_score = len(recalled_communities) / len(golden_communities)
            metric_score_list.append(recall_score)
        
        community_recall = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0

        return {"community_recall": community_recall}, metric_score_list


class ResponseRelevance(BaseMetric):
    """
    响应相关性，评估生成的回答与问题的相关程度
    """
    metric_name = "relevance"

    def __init__(self, config: Dict):
        super().__init__(config)
        self.llm_api_key = config.get('openai_api_key', None)
        self.llm_model = config.get('openai_model', 'gpt-3.5-turbo')
        
        # 如果配置中没有提供API密钥，尝试从环境变量中获取
        if not self.llm_api_key:
            import os
            self.llm_api_key = os.getenv('OPENAI_API_KEY')
            
        # 初始化OpenAI客户端
        self.init_llm_client()

    def init_llm_client(self):
        """初始化LLM客户端"""
        try:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model=self.llm_model,
                api_key=self.llm_api_key,
                temperature=0.0
            )
        except ImportError:
            print("未安装langchain_openai，无法使用ResponseRelevance指标")
            self.llm = None

    def calculate_relevance(self, question: str, answer: str) -> float:
        """
        使用LLM计算问题和回答的相关性
        
        Args:
            question: 问题文本
            answer: 回答文本
            
        Returns:
            相关性分数，范围0-1
        """
        if not self.llm:
            return 0.0
            
        try:
            # 构建评估提示
            prompt = f"""
            问题: {question}
            
            回答: {answer}
            
            请评估上述回答与问题的相关性，给出0到10之间的分数。
            相关性是指回答明确针对问题提供了所需的信息。
            完全不相关的回答得分为0，高度相关且全面的回答得分为10。
            仅输出分数，不要提供解释。
            """
            
            # 使用LLM评估相关性
            result = self.llm.invoke(prompt)
            
            # 提取数字
            content = result.content if hasattr(result, 'content') else str(result)
            score_match = re.search(r'(\d+(\.\d+)?)', content)
            
            if score_match:
                score = float(score_match.group(1))
                # 规范化到0-1范围
                normalized_score = min(1.0, max(0.0, score / 10.0))
                return normalized_score
            else:
                print(f"无法从LLM响应中提取相关性分数: {content}")
                return 0.5  # 默认中等相关性
                
        except Exception as e:
            print(f"计算相关性时出错: {e}")
            return 0.0

    def calculate_metric(self, data: Any) -> Tuple[Dict[str, float], List[float]]:
        """
        计算答案相关性分数
        
        Args:
            data: 包含问题和回答的数据集
            
        Returns:
            tuple: 包含总体相关性分数和每个样本的相关性分数列表
        """
        # 确保数据对象有所需的属性
        if not hasattr(data, 'questions') or not hasattr(data, 'pred'):
            return {"relevance": 0.0}, [0.0] * (len(data.pred) if hasattr(data, 'pred') else 0)
        
        questions = data.questions
        answers = data.pred
        
        metric_score_list = []
        
        for question, answer in zip(questions, answers):
            relevance_score = self.calculate_relevance(question, answer)
            metric_score_list.append(relevance_score)
        
        relevance = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0

        return {"relevance": relevance}, metric_score_list


class ResponseCompleteness(BaseMetric):
    """
    响应完整性，评估生成的回答是否完整解答了问题
    """
    metric_name = "completeness"

    def __init__(self, config: Dict):
        super().__init__(config)
        self.llm_api_key = config.get('openai_api_key', None)
        self.llm_model = config.get('openai_model', 'gpt-3.5-turbo')
        
        # 如果配置中没有提供API密钥，尝试从环境变量中获取
        if not self.llm_api_key:
            import os
            self.llm_api_key = os.getenv('OPENAI_API_KEY')
            
        # 初始化OpenAI客户端
        self.init_llm_client()

    def init_llm_client(self):
        """初始化LLM客户端"""
        try:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model=self.llm_model,
                api_key=self.llm_api_key,
                temperature=0.0
            )
        except ImportError:
            print("未安装langchain_openai，无法使用ResponseCompleteness指标")
            self.llm = None

    def calculate_completeness(self, question: str, answer: str, reference_answer: Optional[str] = None) -> float:
        """
        使用LLM计算回答的完整性
        
        Args:
            question: 问题文本
            answer: 回答文本
            reference_answer: 参考答案 (可选)
            
        Returns:
            完整性分数，范围0-1
        """
        if not self.llm:
            return 0.0
            
        try:
            # 构建评估提示
            if reference_answer:
                prompt = f"""
                问题: {question}
                
                模型回答: {answer}
                
                参考回答: {reference_answer}
                
                请评估模型回答相对于参考回答的完整性，给出0到10之间的分数。
                完整性是指模型回答包含了回答问题所需的全部关键信息。
                完全不完整的回答得分为0，非常完整的回答得分为10。
                仅输出分数，不要提供解释。
                """
            else:
                prompt = f"""
                问题: {question}
                
                回答: {answer}
                
                请评估上述回答的完整性，给出0到10之间的分数。
                完整性是指回答包含了回答问题所需的全部关键信息。
                完全不完整的回答得分为0，非常完整的回答得分为10。
                仅输出分数，不要提供解释。
                """
            
            # 使用LLM评估完整性
            result = self.llm.invoke(prompt)
            
            # 提取数字
            content = result.content if hasattr(result, 'content') else str(result)
            score_match = re.search(r'(\d+(\.\d+)?)', content)
            
            if score_match:
                score = float(score_match.group(1))
                # 规范化到0-1范围
                normalized_score = min(1.0, max(0.0, score / 10.0))
                return normalized_score
            else:
                print(f"无法从LLM响应中提取完整性分数: {content}")
                return 0.5  # 默认中等完整性
                
        except Exception as e:
            print(f"计算完整性时出错: {e}")
            return 0.0

    def calculate_metric(self, data: Any) -> Tuple[Dict[str, float], List[float]]:
        """
        计算答案完整性分数
        
        Args:
            data: 包含问题、回答和参考答案的数据集
            
        Returns:
            tuple: 包含总体完整性分数和每个样本的完整性分数列表
        """
        # 确保数据对象有所需的属性
        if not hasattr(data, 'questions') or not hasattr(data, 'pred'):
            return {"completeness": 0.0}, [0.0] * (len(data.pred) if hasattr(data, 'pred') else 0)
        
        questions = data.questions
        answers = data.pred
        
        # 检查是否有参考答案
        has_references = hasattr(data, 'golden_answers')
        references = data.golden_answers if has_references else None
        
        metric_score_list = []
        
        for i, (question, answer) in enumerate(zip(questions, answers)):
            reference = references[i] if has_references and references else None
            completeness_score = self.calculate_completeness(question, answer, reference)
            metric_score_list.append(completeness_score)
        
        completeness = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0

        return {"completeness": completeness}, metric_score_list