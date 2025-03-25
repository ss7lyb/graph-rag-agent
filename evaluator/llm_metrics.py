import re
from typing import Dict, List, Any, Tuple
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
            entities = sample.retrieved_entities
            relationships = sample.retrieved_relationships
            
            # 准备实体和关系信息
            entities_text = "\n".join([f"- {entity}" for entity in entities[:10]])
            relationships_text = "\n".join([
                f"- {src} --[{rel}]--> {dst}" 
                for src, rel, dst in relationships[:10]
            ])
            
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
            {relationships_text}
            
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