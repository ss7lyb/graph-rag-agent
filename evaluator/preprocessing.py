import re
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
    """从回答中提取引用数据，处理各种格式"""
    entities = []
    relationships = []
    chunks = []
    
    try:
        # 尝试直接提取完整的data部分
        data_match = re.search(r'\{\s*[\'"]?data[\'"]?\s*:\s*\{(.*?)\}\s*\}', answer, re.DOTALL)
        if data_match:
            data_content = data_match.group(1)
            
            # 提取Entities数组
            entities_match = re.search(r'[\'"]?Entities[\'"]?\s*:\s*\[(.*?)\]', data_content, re.DOTALL)
            if entities_match:
                entities_str = entities_match.group(1).strip()
                # 提取所有数字（不带引号）
                number_entities = re.findall(r'\d+', entities_str)
                entities.extend(number_entities)
            
            # 提取Relationships或Reports数组
            rels_match = re.search(r'[\'"]?(?:Relationships|Reports)[\'"]?\s*:\s*\[(.*?)\]', data_content, re.DOTALL)
            if rels_match:
                rels_str = rels_match.group(1).strip()
                # 提取所有数字（不带引号）
                number_relationships = re.findall(r'\d+', rels_str)
                relationships.extend(number_relationships)
            
            # 提取Chunks数组
            chunks_match = re.search(r'[\'"]?Chunks[\'"]?\s*:\s*\[(.*?)\]', data_content, re.DOTALL)
            if chunks_match:
                chunks_str = chunks_match.group(1).strip()
                # 提取引号中的字符串
                quoted_chunks = re.findall(r'[\'"]([^\'"]*)[\'"]', chunks_str)
                chunks.extend(quoted_chunks)
        
        # 如果上面的尝试没有结果，单独查找每个部分
        if not any([entities, relationships, chunks]):
            # 查找 'Entities' 列表
            entities_match = re.search(r'Entities[\'"]?\s*:\s*\[(.*?)\]', answer, re.DOTALL)
            if entities_match:
                entities_str = entities_match.group(1).strip()
                # 提取所有数字（不带引号）
                number_entities = re.findall(r'\b\d+\b', entities_str)
                entities.extend(number_entities)
            
            # 查找 'Relationships' 或 'Reports' 列表
            rels_match = re.search(r'(?:Relationships|Reports)[\'"]?\s*:\s*\[(.*?)\]', answer, re.DOTALL)
            if rels_match:
                rels_str = rels_match.group(1).strip()
                # 提取所有数字（不带引号）
                number_relationships = re.findall(r'\b\d+\b', rels_str)
                relationships.extend(number_relationships)
            
            # 查找 'Chunks' 列表
            chunks_match = re.search(r'Chunks[\'"]?\s*:\s*\[(.*?)\]', answer, re.DOTALL)
            if chunks_match:
                chunks_str = chunks_match.group(1).strip()
                # 提取引号中的字符串
                quoted_chunks = re.findall(r'[\'"]([^\'"]*)[\'"]', chunks_str)
                chunks.extend(quoted_chunks)
        
        # 移除空字符串并去重
        entities = [e for e in entities if e]
        relationships = [r for r in relationships if r]
        chunks = [c for c in chunks if c]
        
        print(f"提取的引用数据：实体={entities}, 关系={relationships}, 文本块={chunks}")
        
        return {
            "entities": entities,
            "relationships": relationships,
            "chunks": chunks
        }
    except Exception as e:
        print(f"提取引用数据失败: {e}")
        return {"entities": [], "relationships": [], "chunks": []}