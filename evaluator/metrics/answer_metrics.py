import re
from typing import Dict, List, Tuple
from evaluator.core.base_metric import BaseMetric
from evaluator.core.evaluation_data import AnswerEvaluationData
from evaluator.utils.text_utils import normalize_answer

class ExactMatch(BaseMetric):
    """精确匹配评估指标"""
    
    metric_name = "em"

    def __init__(self, config):
        super().__init__(config)
        self.llm = config.get("llm", None)
    
    def calculate_em(self, prediction: str, golden_answer: str) -> float:
        """
        计算单个预测的精确匹配得分
        
        Args:
            prediction: 预测答案
            golden_answer: 标准答案
            
        Returns:
            float: 得分（1.0表示匹配，0.0表示不匹配）
        """
        if not prediction or not golden_answer:
            return 0.0
            
        normalized_prediction = normalize_answer(prediction)
        normalized_golden = normalize_answer(golden_answer)
        
        # 完全匹配
        if normalized_prediction == normalized_golden:
            return 1.0
        return 0.0
    
    def calculate_metric(self, data: AnswerEvaluationData) -> Tuple[Dict[str, float], List[float]]:
        """
        计算精确匹配指标 - 使用规则匹配和LLM回退混合评分
        
        Args:
            data: 评估数据
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 总体得分和每个样本的得分
        """
        self.log("======== ExactMatch 计算日志 ========")
        self.log(f"样本总数: {len(data.samples) if hasattr(data, 'samples') else 0}")
        
        golden_answers = data.golden_answers
        system_answers = data.system_answers
        
        metric_score_list = []
        
        for idx, (pred, golden) in enumerate(zip(system_answers, golden_answers)):
            # 预处理系统答案 - 移除Markdown标题和多余空行
            cleaned_pred = re.sub(r'^###.*?\n+', '', pred, flags=re.MULTILINE)
            cleaned_pred = re.sub(r'\n\s*\n', '\n', cleaned_pred)
            cleaned_pred = cleaned_pred.strip()
            
            # 标准化答案
            normalized_pred = normalize_answer(cleaned_pred)
            normalized_golden = normalize_answer(golden)
            
            self.log(f"\n样本 {idx+1}:")
            self.log(f"  标准答案(前30字符): {golden[:30]}...")
            self.log(f"  系统答案(前30字符): {pred[:30]}...")
            self.log(f"  清理后的系统答案(前30字符): {cleaned_pred[:30]}...")
            self.log(f"  标准化后的标准答案(前30字符): {normalized_golden[:30]}...")
            self.log(f"  标准化后的系统答案(前30字符): {normalized_pred[:30]}...")
            
            # 完全匹配
            if normalized_pred == normalized_golden:
                score = 1.0
                self.log(f"  完全匹配 ✓")
            else:
                # 规则匹配失败，回退到LLM评分
                self.log(f"  规则匹配失败，回退到LLM评分")
                
                # 仅当有LLM可用时使用LLM评分
                if self.llm:
                    prompt = f"""
                    请比较下面两个答案，评估它们内容上的等价性，并给出0到1之间的分数。
                    0表示完全不同，1表示内容上完全等价。
                    请只考虑实质内容，忽略格式、表达方式和顺序的差异。
                    
                    标准答案:
                    {golden}
                    
                    系统答案:
                    {cleaned_pred}
                    
                    只返回一个0到1之间的数字表示分数，不要有任何其他文字。
                    """
                    
                    try:
                        response = self.llm.invoke(prompt)
                        score_text = response.content if hasattr(response, 'content') else response
                        
                        self.log(f"  LLM响应: {score_text}")
                        
                        # 提取数字
                        score_match = re.search(r'(\d+(\.\d+)?)', score_text)
                        if score_match:
                            score = float(score_match.group(1))
                            # 确保在0-1范围内
                            score = max(0.0, min(1.0, score))
                            self.log(f"  LLM评估的匹配度分数: {score:.4f}")
                        else:
                            score = 0.0
                            self.log(f"  无法从LLM响应中提取分数，使用默认分数0.0")
                    except Exception as e:
                        self.log(f"  LLM评估时出错: {e}")
                        score = 0.0
                else:
                    # 没有LLM，使用基本规则评分
                    score = 0.0
                    self.log(f"  不匹配且没有可用的LLM评分，使用0分")
                    
                    # 显示不匹配的详细信息
                    min_len = min(len(normalized_pred), len(normalized_golden))
                    max_len = max(len(normalized_pred), len(normalized_golden))
                    
                    # 检查字符级差异
                    first_diff_pos = None
                    for i in range(min_len):
                        if normalized_pred[i] != normalized_golden[i]:
                            first_diff_pos = i
                            break
                    
                    if first_diff_pos is not None:
                        self.log(f"  首个差异位置: 字符位置 {first_diff_pos}")
                        context_start = max(0, first_diff_pos - 10)
                        context_end = min(min_len, first_diff_pos + 10)
                        
                        pred_context = normalized_pred[context_start:context_end]
                        golden_context = normalized_golden[context_start:context_end]
                        
                        self.log(f"  差异上下文 - 系统: '...{pred_context}...'")
                        self.log(f"  差异上下文 - 标准: '...{golden_context}...'")
                    
                    # 显示长度差异
                    if len(normalized_pred) != len(normalized_golden):
                        self.log(f"  答案长度差异: 系统({len(normalized_pred)}) vs 标准({len(normalized_golden)})")
            
            metric_score_list.append(score)
        
        em_score = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0.0
        self.log(f"\n样本总数: {len(metric_score_list)}")
        self.log(f"匹配样本数: {sum(1 for s in metric_score_list if s > 0)}")
        self.log(f"精确匹配平均得分: {em_score:.4f}")
        self.log("======== ExactMatch 计算结束 ========\n")
        
        return {"em": em_score}, metric_score_list

class F1Score(BaseMetric):
    """F1分数评估指标"""
    
    metric_name = "f1"

    def __init__(self, config):
        super().__init__(config)
        self.llm = config.get("llm", None)
    
    def calculate_metric(self, data: AnswerEvaluationData) -> Tuple[Dict[str, float], List[float]]:
        """
        计算F1分数 - 使用规则匹配和LLM回退混合评分
        
        Args:
            data: 评估数据
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 总体得分和每个样本的得分
        """
        self.log("\n======== F1Score 计算日志 ========")
        self.log(f"样本总数: {len(data.samples) if hasattr(data, 'samples') else 0}")
        
        golden_answers = data.golden_answers
        system_answers = data.system_answers
        
        f1_scores = []
        
        for idx, (pred, golden) in enumerate(zip(system_answers, golden_answers)):
            # 预处理系统答案 - 移除Markdown标题和多余空行
            cleaned_pred = re.sub(r'^###.*?\n+', '', pred, flags=re.MULTILINE)
            cleaned_pred = re.sub(r'\n\s*\n', '\n', cleaned_pred)
            cleaned_pred = cleaned_pred.strip()
            
            # 将文本标准化
            pred_text = normalize_answer(cleaned_pred)
            golden_text = normalize_answer(golden)
            
            self.log(f"\n样本 {idx+1}:")
            self.log(f"  标准答案(前30字符): {golden[:30]}...")
            self.log(f"  系统答案(前30字符): {pred[:30]}...")
            
            # 尝试使用传统F1计算
            try:
                # 进行中文分词
                import jieba
                pred_tokens = list(jieba.cut(pred_text))
                golden_tokens = list(jieba.cut(golden_text))
                
                # 过滤停用词和过短的词
                stopwords = {'的', '了', '和', '在', '是', '为', '以', '与', '或', '且'}
                pred_tokens = [token for token in pred_tokens if len(token) > 1 and token not in stopwords]
                golden_tokens = [token for token in golden_tokens if len(token) > 1 and token not in stopwords]
                
                self.log(f"  标准答案分词数: {len(golden_tokens)}")
                self.log(f"  系统答案分词数: {len(pred_tokens)}")
                
                if not pred_tokens or not golden_tokens:
                    # 空文本处理
                    if not pred_tokens and not golden_tokens:
                        rule_f1 = 1.0  # 两者都为空，视为完全匹配
                        self.log(f"  两者都为空，视为完全匹配，F1=1.0")
                    else:
                        rule_f1 = 0.0  # 一个为空一个不为空
                        self.log(f"  一个为空一个不为空，规则F1=0.0")
                else:
                    # 计算标准F1
                    common_tokens = set(pred_tokens) & set(golden_tokens)
                    precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
                    recall = len(common_tokens) / len(golden_tokens) if golden_tokens else 0
                    
                    if precision + recall > 0:
                        rule_f1 = 2 * precision * recall / (precision + recall)
                    else:
                        rule_f1 = 0.0
                    
                    self.log(f"  共有词汇: {len(common_tokens)}/{len(set(pred_tokens) | set(golden_tokens))}")
                    self.log(f"  精确率: {precision:.4f}")
                    self.log(f"  召回率: {recall:.4f}")
                    self.log(f"  规则F1分数: {rule_f1:.4f}")
            except Exception as e:
                self.log(f"  规则F1计算出错: {e}")
                rule_f1 = 0.0
            
            # 如果规则F1太低，回退到LLM评分
            if rule_f1 < 0.3 and self.llm:
                self.log(f"  规则F1分数过低 ({rule_f1:.4f})，回退到LLM评分")
                
                # 构建内容相似度评估提示
                prompt = f"""
                请比较下面两个答案的内容相似度，评估它们包含的信息重叠程度，并给出0到1之间的分数。
                0表示完全不同信息，1表示信息完全重叠。
                请考虑实质内容的相似性，而不仅是表面文字的匹配。
                
                标准答案:
                {golden}
                
                系统答案:
                {cleaned_pred}
                
                只返回一个0到1之间的数字表示分数，不要有任何其他文字。
                """
                
                try:
                    response = self.llm.invoke(prompt)
                    score_text = response.content if hasattr(response, 'content') else response
                    
                    self.log(f"  LLM响应: {score_text}")
                    
                    # 提取数字
                    score_match = re.search(r'(\d+(\.\d+)?)', score_text)
                    if score_match:
                        llm_f1 = float(score_match.group(1))
                        # 确保在0-1范围内
                        llm_f1 = max(0.0, min(1.0, llm_f1))
                        self.log(f"  LLM评估的F1分数: {llm_f1:.4f}")
                        
                        # 使用LLM的分数
                        f1 = llm_f1
                    else:
                        self.log(f"  无法从LLM响应中提取分数，使用规则F1分数")
                        f1 = rule_f1
                except Exception as e:
                    self.log(f"  LLM评估时出错: {e}")
                    f1 = rule_f1
            else:
                # 规则F1分数足够好，或者没有LLM可用
                f1 = rule_f1
            
            f1_scores.append(f1)
        
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        
        self.log(f"\n样本总数: {len(f1_scores)}")
        self.log(f"F1得分大于0.5的样本数: {sum(1 for s in f1_scores if s > 0.5)}")
        self.log(f"F1平均得分: {avg_f1:.4f}")
        self.log("======== F1Score 计算结束 ========\n")
        
        return {"f1": avg_f1}, f1_scores