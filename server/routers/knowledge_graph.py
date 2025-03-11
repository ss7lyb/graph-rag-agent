from fastapi import APIRouter
from typing import Optional
from services.kg_service import (
    get_knowledge_graph, 
    extract_kg_from_message, 
    get_entity_types, 
    get_chunks
)

# 创建路由器
router = APIRouter()


@router.get("/knowledge_graph")
async def knowledge_graph(limit: int = 100, query: Optional[str] = None):
    """
    获取知识图谱数据
    
    Args:
        limit: 节点数量限制
        query: 查询条件(可选)
        
    Returns:
        Dict: 知识图谱数据，包含节点和连接
    """
    return get_knowledge_graph(limit, query)


@router.get("/knowledge_graph_from_message")
async def knowledge_graph_from_message(message: Optional[str] = None, query: Optional[str] = None):
    """
    从消息文本中提取知识图谱数据
    
    Args:
        message: 消息文本
        query: 查询内容(可选)
        
    Returns:
        Dict: 知识图谱数据，包含节点和连接
    """
    if not message:
        return {"nodes": [], "links": []}
    
    return extract_kg_from_message(message, query)


@router.get("/entity_types")
async def entity_types():
    """
    获取数据库中所有可能的实体类型
    
    Returns:
        Dict: 包含实体类型列表
    """
    types = get_entity_types()
    return {"entity_types": types}


@router.get("/chunks")
async def chunks(limit: int = 10, offset: int = 0):
    """
    获取数据库中的文本块
    
    Args:
        limit: 返回数量限制
        offset: 偏移量
        
    Returns:
        Dict: 文本块数据和总数
    """
    return get_chunks(limit, offset)