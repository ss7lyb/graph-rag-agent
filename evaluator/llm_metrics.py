import re
from typing import Dict, List, Any, Tuple
import json

from evaluator.base import BaseMetric

class ResponseCoherence(BaseMetric):
    """
    回答连贯性评估指标 - 评估回答的连贯性和结构化程度
    """
    
    metric_name = "response_coherence"
    
    def __init__(self, config):
        super().__init__(config)
        self.llm = config.get("llm", None)
    
    def calculate_metric(self, data) -> Tuple[Dict[str, float], List[float]]:
        """
        计算回答连贯性
        
        Args:
            data: 评估数据
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 总体得分和每个样本的得分
        """
        if not self.llm:
            return {"response_coherence": 0.0}, [0.0] * len(data.samples)
        
        coherence_scores = []
        
        for sample in data.samples:
            question = sample.question
            answer = sample.system_answer
            
            # 使用LLM评估连贯性
            prompt = f"""
            评估以下回答的连贯性和结构，给出0到1的分数。
            评分标准:
            - 高分(0.8-1.0): 逻辑清晰，结构良好，使用标题和段落，思路连贯
            - 中分(0.4-0.7): 内容基本清晰，但可能存在一些逻辑跳跃
            - 低分(0.0-0.3): 结构混乱，缺乏逻辑性
            
            问题: {question}
            回答: {answer}
            
            只返回一个0到1之间的数字表示分数，不要有任何其他文字。
            """
            
            try:
                response = self.llm.invoke(prompt)
                score_text = response.content if hasattr(response, 'content') else response
                
                # 提取数字
                score_match = re.search(r'(\d+(\.\d+)?)', score_text)
                if score_match:
                    coherence = float(score_match.group(1))
                    # 确保在0-1范围内
                    coherence = max(0.0, min(1.0, coherence))
                else:
                    coherence = 0.5  # 默认中等分数
            except Exception as e:
                print(f"LLM评估连贯性时出错: {e}")
                coherence = 0.5
                
            coherence_scores.append(coherence)
        
        avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0
        
        return {"response_coherence": avg_coherence}, coherence_scores


class FactualConsistency(BaseMetric):
    """
    事实一致性评估指标 - 评估回答与检索到的事实的一致性
    """
    
    metric_name = "factual_consistency"
    
    def __init__(self, config):
        super().__init__(config)
        self.llm = config.get("llm", None)
    
    def calculate_metric(self, data) -> Tuple[Dict[str, float], List[float]]:
        """
        计算事实一致性
        
        Args:
            data: 评估数据
                
        Returns:
            Tuple[Dict[str, float], List[float]]: 总体得分和每个样本的得分
        """
        if not self.llm:
            return {"factual_consistency": 0.0}, [0.0] * len(data.samples)
        
        consistency_scores = []
        
        for sample in data.samples:
            answer = sample.system_answer
            
            # 检查是否有检索实体和关系属性，这是RetrievalEvaluationSample特有的
            if hasattr(sample, 'retrieved_entities'):
                entities = sample.retrieved_entities
                relationships = sample.retrieved_relationships
            else:
                # 对于AnswerEvaluationSample，可以从引用数据中提取
                from evaluator.preprocessing import extract_references_from_answer
                refs = extract_references_from_answer(answer)
                entities = refs.get("entities", [])
                relationships = refs.get("relationships", [])
            
            # 准备实体和关系信息
            entities_text = "\n".join([f"- {entity}" for entity in entities[:10]])
            
            # 处理关系，确保它们是三元组格式
            relations_text = ""
            if relationships:
                # 处理不同格式的关系
                formatted_relations = []
                for rel in relationships[:10]:
                    if isinstance(rel, tuple) and len(rel) >= 3:
                        formatted_relations.append(f"- {rel[0]} --[{rel[1]}]--> {rel[2]}")
                    elif isinstance(rel, list) and len(rel) >= 3:
                        formatted_relations.append(f"- {rel[0]} --[{rel[1]}]--> {rel[2]}")
                    elif isinstance(rel, str):
                        formatted_relations.append(f"- 关系ID: {rel}")
                
                relations_text = "\n".join(formatted_relations)
            
            # 使用LLM评估事实一致性
            prompt = f"""
            评估以下回答与检索到的图数据保持一致的程度，给出0到1的分数。
            评分标准:
            - 高分(0.8-1.0): 回答中的信息完全由图数据支持，没有添加未见的信息
            - 中分(0.4-0.7): 回答中部分信息有图数据支持，但包含一些无法验证的内容
            - 低分(0.0-0.3): 回答与图数据不一致或添加了大量未见的信息
            
            检索到的实体:
            {entities_text}
            
            检索到的关系:
            {relations_text}
            
            回答:
            {answer}
            
            只返回一个0到1之间的数字表示分数，不要有任何其他文字。
            """
            
            try:
                response = self.llm.invoke(prompt)
                score_text = response.content if hasattr(response, 'content') else response
                
                # 提取数字
                score_match = re.search(r'(\d+(\.\d+)?)', score_text)
                if score_match:
                    consistency = float(score_match.group(1))
                    # 确保在0-1范围内
                    consistency = max(0.0, min(1.0, consistency))
                else:
                    consistency = 0.5  # 默认中等分数
            except Exception as e:
                print(f"LLM评估事实一致性时出错: {e}")
                consistency = 0.5
                    
            consistency_scores.append(consistency)
        
        avg_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0
        
        return {"factual_consistency": avg_consistency}, consistency_scores


class ComprehensiveAnswerMetric(BaseMetric):
    """
    回答全面性评估指标 - 评估回答是否全面解答了问题
    """
    
    metric_name = "answer_comprehensiveness"
    
    def __init__(self, config):
        super().__init__(config)
        self.llm = config.get("llm", None)
    
    def calculate_metric(self, data) -> Tuple[Dict[str, float], List[float]]:
        """
        计算回答全面性
        
        Args:
            data: 评估数据
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 总体得分和每个样本的得分
        """
        if not self.llm:
            return {"answer_comprehensiveness": 0.0}, [0.0] * len(data.samples)
        
        comprehensiveness_scores = []
        
        for sample in data.samples:
            question = sample.question
            answer = sample.system_answer
            
            # 使用LLM评估全面性
            prompt = f"""
            评估以下回答解决问题的全面性，给出0到1的分数。
            评分标准:
            - 高分(0.8-1.0): 回答全面地解决了问题的所有方面，提供了丰富的信息和细节
            - 中分(0.4-0.7): 回答基本解决了问题，但可能遗漏了一些次要方面
            - 低分(0.0-0.3): 回答不完整，忽略了问题的主要方面
            
            问题: {question}
            回答: {answer}
            
            只返回一个0到1之间的数字表示分数，不要有任何其他文字。
            """
            
            try:
                response = self.llm.invoke(prompt)
                score_text = response.content if hasattr(response, 'content') else response
                
                # 提取数字
                score_match = re.search(r'(\d+(\.\d+)?)', score_text)
                if score_match:
                    comprehensiveness = float(score_match.group(1))
                    # 确保在0-1范围内
                    comprehensiveness = max(0.0, min(1.0, comprehensiveness))
                else:
                    comprehensiveness = 0.5  # 默认中等分数
            except Exception as e:
                print(f"LLM评估全面性时出错: {e}")
                comprehensiveness = 0.5
                
            comprehensiveness_scores.append(comprehensiveness)
        
        avg_comprehensiveness = sum(comprehensiveness_scores) / len(comprehensiveness_scores) if comprehensiveness_scores else 0.0
        
        return {"answer_comprehensiveness": avg_comprehensiveness}, comprehensiveness_scores

class LLMGraphRagEvaluator(BaseMetric):
    """
    使用LLM评估GraphRAG和HybridRAG的性能
    """
    
    metric_name = "llm_evaluation"
    
    def __init__(self, config):
        super().__init__(config)
        self.llm = config.get("llm", None)
        self.aspect_weights = {
            "comprehensiveness": 0.3,  # 全面性
            "relativeness": 0.25,      # 相关性 
            "empowerment": 0.25,       # 增强理解能力
            "directness": 0.2          # 直接性
        }
        
        # 如果没有提供LLM，则无法执行评估
        if not self.llm:
            print("警告: 未提供LLM模型，无法执行LLM评估")
    
    def calculate_metric(self, data) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        使用LLM计算评估指标
        
        Args:
            data: 评估数据
            
        Returns:
            Tuple[Dict[str, float], List[Dict[str, float]]]: 总体得分和每个样本的得分
        """
        if not self.llm:
            empty_scores = {f"llm_{aspect}": 0.0 for aspect in self.aspect_weights}
            return empty_scores, [{} for _ in data.samples]
        
        all_scores = []
        summary_scores = {aspect: [] for aspect in self.aspect_weights}
        
        for sample in data.samples:
            question = sample.question
            answer = sample.system_answer
            
            # 执行评估
            sample_scores = self._evaluate_answer(question, answer)
            all_scores.append(sample_scores)
            
            # 更新每个指标的累积分数
            for aspect, score in sample_scores.items():
                if aspect in summary_scores:
                    summary_scores[aspect].append(score)
        
        # 计算平均分数
        avg_scores = {}
        for aspect, scores in summary_scores.items():
            if scores:
                avg_scores[f"llm_{aspect}"] = sum(scores) / len(scores)
            else:
                avg_scores[f"llm_{aspect}"] = 0.0
        
        # 计算加权总分
        weighted_sum = sum(avg_scores[f"llm_{aspect}"] * weight 
                         for aspect, weight in self.aspect_weights.items())
        avg_scores["llm_total"] = weighted_sum
        
        return avg_scores, all_scores
    
    def _evaluate_answer(self, question: str, answer: str) -> Dict[str, float]:
        """
        对单个回答进行评估
        
        Args:
            question: 问题
            answer: 回答
            
        Returns:
            Dict[str, float]: 各个方面的评分
        """
        # 清理回答，移除引用数据部分
        cleaned_answer = self._clean_references(answer)
        
        # 使用LLM评估各个方面
        eval_prompt = self._create_evaluation_prompt(question, cleaned_answer)
        
        try:
            response = self.llm.invoke(eval_prompt)
            content = response.content if hasattr(response, 'content') else response
            
            # 解析评估结果
            return self._parse_evaluation_result(content)
        except Exception as e:
            print(f"LLM评估出错: {e}")
            return {aspect: 0.5 for aspect in self.aspect_weights}  # 默认中等分数
    
    def _clean_references(self, answer: str) -> str:
        """清理引用数据部分"""
        # 移除引用数据部分
        cleaned = re.sub(r'#{1,4}\s*引用数据[\s\S]*?(\{[\s\S]*?\})\s*$', '', answer)
        
        # 如果没有引用数据部分，尝试其他格式
        if cleaned == answer:
            cleaned = re.sub(r'#### 引用数据[\s\S]*?(\{[\s\S]*?\})\s*$', '', answer)
        
        # 移除任何尾部空行
        cleaned = cleaned.rstrip()
        
        return cleaned
    
    def _create_evaluation_prompt(self, question: str, answer: str) -> str:
        """创建用于评估的提示"""
        return f"""
        请评估以下回答相对于问题的质量，给出0到1之间的分数（可以使用小数）。
        
        评估应该从以下四个方面进行：
        
        1. 全面性(comprehensiveness)：回答涵盖了问题的各个方面的程度
           - 0分表示完全不全面，遗漏重要信息
           - 1分表示非常全面，涵盖所有相关内容
        
        2. 相关性(relativeness)：回答与问题的相关程度
           - 0分表示完全不相关
           - 1分表示高度相关，直接回应问题
        
        3. 增强理解能力(empowerment)：回答帮助读者理解并做出判断的程度
           - 0分表示没有帮助理解
           - 1分表示显著增强了理解
        
        4. 直接性(directness)：回答直接回应问题，不偏离主题的程度
           - 0分表示完全间接，偏离主题
           - 1分表示直接明了，切中要点
        
        问题: {question}
        
        回答: {answer}
        
        请以JSON格式返回评分结果，格式为：
        {{
            "comprehensiveness": 0.X,
            "relativeness": 0.X,
            "empowerment": 0.X,
            "directness": 0.X,
            "reasoning": "简短解释评分理由"
        }}
        
        只返回JSON对象，不要有任何其他文字。
        """
    
    def _parse_evaluation_result(self, content: str) -> Dict[str, float]:
        """解析LLM的评估结果"""
        # 尝试提取JSON部分
        json_match = re.search(r'(\{[\s\S]*\})', content)
        if not json_match:
            return {aspect: 0.5 for aspect in self.aspect_weights}
        
        try:
            json_str = json_match.group(1)
            data = json.loads(json_str)
            
            # 提取评分
            scores = {}
            for aspect in self.aspect_weights:
                if aspect in data and isinstance(data[aspect], (int, float)):
                    scores[aspect] = min(1.0, max(0.0, float(data[aspect])))
                else:
                    scores[aspect] = 0.5  # 默认中等分数
            
            return scores
        except Exception as e:
            print(f"解析LLM评估结果出错: {e}")
            return {aspect: 0.5 for aspect in self.aspect_weights}