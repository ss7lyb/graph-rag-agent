from typing import Dict

class AnswerValidator:
    """
    答案验证器：评估生成答案的质量，确保满足基本要求
    """
    
    def __init__(self, keyword_extractor=None):
        """
        初始化验证器
        
        参数:
            keyword_extractor: 用于提取关键词的函数或对象
        """
        self.keyword_extractor = keyword_extractor
        self.error_patterns = [
            "抱歉，处理您的问题时遇到了错误",
            "技术原因:",
            "无法获取",
            "无法回答这个问题",
            "没有找到相关信息",
            "对不起，我不能"
        ]
    
    def validate(self, query: str, answer: str) -> Dict[str, bool]:
        """
        验证生成答案的质量
        
        参数:
            query: 原始查询
            answer: 生成的答案
            
        返回:
            Dict[str, bool]: 各项验证的结果
        """
        results = {}
        
        # 检查最小长度
        results["length"] = len(answer) >= 50
        if not results["length"]:
            print(f"[验证] 答案太短: {len(answer)}字符")
        
        # 检查是否包含错误模式
        results["no_error_patterns"] = not any(pattern in answer for pattern in self.error_patterns)
        if not results["no_error_patterns"]:
            for pattern in self.error_patterns:
                if pattern in answer:
                    print(f"[验证] 答案包含错误模式: {pattern}")
                    break
        
        # 关键词相关性检查
        results["keyword_relevance"] = self._check_keyword_relevance(query, answer)
        
        # 总体通过验证
        results["passed"] = all(results.values())
        
        return results
    
    def _check_keyword_relevance(self, query: str, answer: str) -> bool:
        """
        检查答案是否包含查询的关键词
        
        参数:
            query: 查询字符串
            answer: 生成的答案
            
        返回:
            bool: 是否满足关键词相关性要求
        """
        # 如果没有关键词提取器，则默认通过
        if not self.keyword_extractor:
            return True
            
        # 提取关键词
        keywords = self.keyword_extractor(query)
        if not keywords:
            return True
            
        high_level_keywords = keywords.get("high_level", [])
        low_level_keywords = keywords.get("low_level", [])
        
        # 至少有一个高级关键词应该在答案中出现
        if high_level_keywords:
            keyword_found = any(keyword.lower() in answer.lower() for keyword in high_level_keywords)
            if not keyword_found:
                print(f"[验证] 答案未包含任何高级关键词: {high_level_keywords}")
                return False
                
        # 至少有一半的低级关键词应该在答案中出现
        if low_level_keywords and len(low_level_keywords) > 1:
            matches = sum(1 for keyword in low_level_keywords if keyword.lower() in answer.lower())
            if matches < len(low_level_keywords) / 2:
                print(f"[验证] 答案未包含足够的低级关键词: {matches}/{len(low_level_keywords)}")
                return False
        
        print("[验证] 答案通过关键词相关性检查")
        return True