import re
import string
from typing import List, Dict, Any, Set, Tuple
import json

def normalize_answer(s: str) -> str:
    """
    标准化答案文本，移除冠词、标点符号，转为小写，修复空格
    
    Args:
        s (str): 原始文本
        
    Returns:
        str: 标准化后的文本
    """
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the|一个|一种|这个|那个)\b", " ", text)
    
    def white_space_fix(text: str) -> str:
        return " ".join(text.split())
    
    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation + "，。！？《》【】""''：；（）、")
        return "".join(ch for ch in text if ch not in exclude)
    
    def lower(text: str) -> str:
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def extract_entities_from_neo4j_response(text: str) -> List[str]:
    """
    从Neo4j响应中提取实体名称
    
    Args:
        text (str): Neo4j响应文本
        
    Returns:
        List[str]: 实体名称列表
    """
    # 尝试提取实体名称
    entity_pattern = r'(?:节点|实体|Node|Entities|Entity)[:：]\s*["\'](.*?)["\']'
    entities = re.findall(entity_pattern, text, re.IGNORECASE)
    
    # 去重
    return list(set(entities))

def extract_relationships_from_neo4j_response(text: str) -> List[Tuple[str, str, str]]:
    """
    从Neo4j响应中提取关系信息
    
    Args:
        text (str): Neo4j响应文本
        
    Returns:
        List[Tuple[str, str, str]]: 关系列表，每个元素为(源实体, 关系类型, 目标实体)
    """
    # 尝试提取结构化的关系信息
    rel_pattern = r'(?:["\'](.*?)["\'])\s*(?:和|与)\s*(?:["\'](.*?)["\'])\s*(?:之间|存在|有)\s*(?:["\'](.*?)["\'])\s*(?:关系|联系)'
    relationships = re.findall(rel_pattern, text, re.IGNORECASE)
    
    # 如果没找到，尝试其他格式
    if not relationships:
        rel_pattern = r'(?:["\'](.*?)["\'])\s*(?:["\'](.*?)["\'])\s*(?:["\'](.*?)["\'])'
        relationships = re.findall(rel_pattern, text, re.IGNORECASE)
    
    # 去重
    seen = set()
    unique_rels = []
    for rel in relationships:
        if rel not in seen:
            seen.add(rel)
            unique_rels.append(rel)
    
    return unique_rels

def extract_json_from_text(text: str) -> Any:
    """
    从文本中提取JSON内容
    
    Args:
        text (str): 包含JSON的文本
        
    Returns:
        Any: 解析后的JSON对象
    """
    # 寻找可能的JSON内容
    json_pattern = r'{.*}'
    json_match = re.search(json_pattern, text, re.DOTALL)
    
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # 尝试修复常见的JSON格式错误
            # 单引号替换为双引号
            json_str = json_str.replace("'", '"')
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
    
    return None

def compute_overlap_score(pred: List[str], truth: List[str]) -> float:
    """
    计算两个列表的重叠分数
    
    Args:
        pred (List[str]): 预测列表
        truth (List[str]): 真实列表
        
    Returns:
        float: Jaccard相似度 (0-1之间)
    """
    if not pred or not truth:
        return 0.0
    
    # 标准化处理
    pred_norm = [normalize_answer(p) for p in pred]
    truth_norm = [normalize_answer(t) for t in truth]
    
    # 计算交集大小
    intersection = set(pred_norm).intersection(set(truth_norm))
    
    # 计算并集大小
    union = set(pred_norm).union(set(truth_norm))
    
    # Jaccard相似度
    if len(union) == 0:
        return 0.0
    
    return len(intersection) / len(union)

def compute_precision_recall_f1(pred: List[str], truth: List[str]) -> Dict[str, float]:
    """
    计算精确率、召回率、F1分数
    
    Args:
        pred (List[str]): 预测列表
        truth (List[str]): 真实列表
        
    Returns:
        Dict[str, float]: 包含precision, recall, f1的字典
    """
    if not pred or not truth:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # 标准化处理
    pred_norm = [normalize_answer(p) for p in pred]
    truth_norm = [normalize_answer(t) for t in truth]
    
    # 计算交集大小
    tp = len(set(pred_norm).intersection(set(truth_norm)))
    
    # 计算精确率和召回率
    precision = tp / len(pred_norm) if pred_norm else 0.0
    recall = tp / len(truth_norm) if truth_norm else 0.0
    
    # 计算F1分数
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision, 
        "recall": recall, 
        "f1": f1
    }

def extract_referenced_entities(answer: str) -> Set[str]:
    """
    从回答中提取引用的实体
    
    Args:
        answer (str): 回答文本
        
    Returns:
        Set[str]: 实体集合
    """
    # 尝试提取实体名称
    patterns = [
        r'(?:["\'](.*?)["\'](?:\s+(?:是|表示|代表|指|指代)))',
        r'(?:(?:实体|节点|Node)[:：]\s*["\'](.*?)["\'])',
        r'(?:\b([\w\u4e00-\u9fa5]+)\b(?:\s+是|表示|被称为|指代))'
    ]
    
    entities = set()
    for pattern in patterns:
        matches = re.findall(pattern, answer, re.IGNORECASE)
        entities.update(matches)
    
    # 过滤空字符串和太长的内容（可能是误匹配）
    return {e for e in entities if e and len(e) < 50}

def extract_referenced_relationships(answer: str) -> List[Tuple[str, str, str]]:
    """
    从回答中提取引用的关系
    
    Args:
        answer (str): 回答文本
        
    Returns:
        List[Tuple[str, str, str]]: 关系列表，每个元素为(源实体, 关系类型, 目标实体)
    """
    # 尝试提取关系
    patterns = [
        r'(?:["\'](.*?)["\'])\s*(?:和|与)\s*(?:["\'](.*?)["\'])\s*(?:之间|存在|有)\s*(?:["\'](.*?)["\'])\s*(?:关系|联系)',
        r'(?:["\'](.*?)["\'])\s*(?:["\'](.*?)["\'])\s*(?:["\'](.*?)["\'])',
        r'(?:([\w\u4e00-\u9fa5]+))\s*(?:和|与)\s*(?:([\w\u4e00-\u9fa5]+))\s*(?:之间|存在|有)\s*(?:([\w\u4e00-\u9fa5]+))'
    ]
    
    relationships = []
    for pattern in patterns:
        matches = re.findall(pattern, answer, re.IGNORECASE)
        relationships.extend(matches)
    
    # 过滤可能的误匹配
    valid_relationships = []
    for src, rel, dst in relationships:
        if src and rel and dst and len(src) < 50 and len(rel) < 50 and len(dst) < 50:
            valid_relationships.append((src, rel, dst))
    
    return valid_relationships
