import re
import json
from typing import Dict, Any

def clean_references(answer: str) -> str:
    """
    清理AI回答中的引用数据部分
    
    参数:
        answer: AI生成的回答
        
    返回:
        清理后的回答
    """
    # 移除引用数据部分
    cleaned = re.sub(r'###\s*引用数据[\s\S]*?(\{\s*[\'"]data[\'"][\s\S]*?\}\s*)', '', answer)
    
    # 如果没有引用数据部分，尝试其他格式
    if cleaned == answer:
        cleaned = re.sub(r'#### 引用数据[\s\S]*?(\{\s*[\'"]data[\'"][\s\S]*?\}\s*)', '', answer)
    
    # 移除任何尾部空行
    cleaned = cleaned.rstrip()
    
    return cleaned

def clean_thinking_process(answer: str) -> str:
    """
    清理deep agent回答中的思考过程
    
    参数:
        answer: AI生成的回答
        
    返回:
        清理后的回答，没有思考过程
    """
    # 移除思考过程部分
    cleaned = re.sub(r'<think>[\s\S]*?</think>\s*', '', answer)
    
    # 移除任何多余的空行
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned

def extract_references_from_answer(answer: str) -> Dict[str, Any]:
    """
    从回答中提取引用数据
    
    参数:
        answer: AI生成的回答
        
    返回:
        包含实体和关系的字典
    """
    # 寻找引用数据部分
    refs_match = re.search(r'(?:###|####)\s*引用数据[\s\S]*?(\{\s*[\'"]data[\'"][\s\S]*?\}\s*)', answer)
    
    if not refs_match:
        return {"entities": [], "relationships": [], "chunks": []}
    
    # 提取JSON部分
    try:
        json_str = refs_match.group(1).replace("'", '"')
        refs_data = json.loads(json_str)
        
        entities = []
        relationships = []
        chunks = []
        
        # 提取实体
        if "data" in refs_data:
            data = refs_data["data"]
            entities.extend(data.get("Entities", []))
            relationships.extend(data.get("Relationships", []))
            chunks.extend(data.get("Chunks", []))
            
        return {
            "entities": entities, 
            "relationships": relationships,
            "chunks": chunks
        }
    except Exception as e:
        print(f"提取引用数据失败: {e}")
        return {"entities": [], "relationships": [], "chunks": []}