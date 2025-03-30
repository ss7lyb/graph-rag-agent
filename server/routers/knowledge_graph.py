from fastapi import APIRouter
from typing import Optional
import traceback
from services.kg_service import (
    get_knowledge_graph, 
    extract_kg_from_message, 
    get_entity_types, 
    get_chunks,
    get_relation_types,
    get_shortest_path,
    get_one_two_hop_paths,
    get_common_neighbors,
    get_all_paths,
    get_entity_cycles,
    get_entity_influence,
    get_simplified_community,
)
from server_config.database import get_db_manager
from models.schemas import ReasoningRequest

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

@router.post("/kg_reasoning")
async def knowledge_graph_reasoning(request: ReasoningRequest):
    """
    执行知识图谱推理
    """
    try:
        # 获取数据库连接
        db_manager = get_db_manager()
        driver = db_manager.get_driver()
        
        # 对参数进行处理，确保安全传递给Neo4j
        reasoning_type = request.reasoning_type
        entity_a = request.entity_a.strip()
        entity_b = request.entity_b.strip() if request.entity_b else None
        max_depth = min(max(1, request.max_depth), 5)  # 确保在1-5的范围内
        algorithm = request.algorithm
        
        print(f"推理请求: 类型={reasoning_type}, 实体A={entity_a}, 实体B={entity_b}, 深度={max_depth}, 算法={algorithm}")
        
        # 社区检测系统
        if reasoning_type == "entity_community":
            return await process_community_detection(entity_a, max_depth, algorithm)
            
        # 其他推理类型
        if reasoning_type == "shortest_path":
            if not entity_b:
                return {"error": "最短路径查询需要指定两个实体", "nodes": [], "links": []}
            result = get_shortest_path(driver, entity_a, entity_b, max_depth)
        elif reasoning_type == "one_two_hop":
            if not entity_b:
                return {"error": "一到两跳关系查询需要指定两个实体", "nodes": [], "links": []}
            result = get_one_two_hop_paths(driver, entity_a, entity_b)
        elif reasoning_type == "common_neighbors":
            if not entity_b:
                return {"error": "共同邻居查询需要指定两个实体", "nodes": [], "links": []}
            result = get_common_neighbors(driver, entity_a, entity_b)
        elif reasoning_type == "all_paths":
            if not entity_b:
                return {"error": "关系路径查询需要指定两个实体", "nodes": [], "links": []}
            result = get_all_paths(driver, entity_a, entity_b, max_depth)
        elif reasoning_type == "entity_cycles":
            result = get_entity_cycles(driver, entity_a, max_depth)
        elif reasoning_type == "entity_influence":
            result = get_entity_influence(driver, entity_a, max_depth)
        else:
            return {"error": "未知的推理类型", "nodes": [], "links": []}
        
        return result
    except Exception as e:
        print(f"推理查询异常: {str(e)}")
        traceback.print_exc()
        return {"error": str(e), "nodes": [], "links": []}

async def process_community_detection(entity_id: str, max_depth: int, algorithm: str):
    """执行专业社区检测流程"""
    try:
        # 首先检查实体是否已存在于社区中
        community_info = await get_entity_community_from_db(entity_id)
        if community_info and community_info.get("nodes") and community_info.get("links"):
            print(f"实体 {entity_id} 已有社区信息，直接返回")
            return community_info
            
        # 实体没有社区信息，使用简化版本返回查询结果
        print(f"实体 {entity_id} 没有社区信息，使用简化版本")
        db_manager = get_db_manager()
        driver = db_manager.get_driver()
        return get_simplified_community(driver, entity_id, max_depth)
    except Exception as e:
        print(f"处理社区检测失败: {str(e)}")
        traceback.print_exc()
        return {"error": str(e), "nodes": [], "links": []}

async def get_entity_community_from_db(entity_id: str):
    """从数据库中获取实体的社区信息"""
    try:
        db_manager = get_db_manager()
        graph = db_manager.get_graph()
        
        # 查询实体所属的社区
        community_result = graph.query("""
        MATCH (e:__Entity__ {id: $entity_id})-[:IN_COMMUNITY]->(c:__Community__)
        RETURN c.id AS community_id
        LIMIT 1
        """, params={"entity_id": entity_id})
        
        if not community_result:
            return None
            
        community_id = community_result[0].get("community_id")
        if not community_id:
            return None
            
        # 获取该社区的所有节点和关系
        community_data = graph.query("""
        // 获取社区中的所有实体
        MATCH (c:__Community__ {id: $community_id})<-[:IN_COMMUNITY]-(e:__Entity__)
        WITH c, collect({
            id: e.id,
            description: e.description,
            labels: labels(e)
        }) AS entities
        
        // 获取社区摘要
        OPTIONAL MATCH (c)
        WHERE c.summary IS NOT NULL
        
        // 获取实体间的关系
        CALL {
            WITH c
            MATCH (c)<-[:IN_COMMUNITY]-(e1:__Entity__)-[r]->(e2:__Entity__)-[:IN_COMMUNITY]->(c)
            RETURN collect({
                source: e1.id,
                target: e2.id,
                type: type(r)
            }) AS relationships
        }
        
        // 返回社区信息
        RETURN 
            c.id AS community_id,
            c.summary AS summary,
            entities,
            relationships
        """, params={"community_id": community_id})
        
        if not community_data:
            return None
            
        # 构建可视化格式
        nodes = []
        links = []
        community_summary = community_data[0].get("summary", "无社区摘要")
        
        # 处理节点
        for entity in community_data[0].get("entities", []):
            entity_labels = entity.get("labels", [])
            group = [label for label in entity_labels if label != "__Entity__"]
            group = group[0] if group else "Unknown"
            
            # 标记中心实体
            if entity.get("id") == entity_id:
                group = "Center"
                
            nodes.append({
                "id": entity.get("id"),
                "label": entity.get("id"),
                "description": entity.get("description", ""),
                "group": group
            })
        
        # 处理关系
        for rel in community_data[0].get("relationships", []):
            links.append({
                "source": rel.get("source"),
                "target": rel.get("target"),
                "label": rel.get("type"),
                "weight": 1
            })
        
        # 获取社区统计信息
        stats = {
            "id": community_id,
            "entity_count": len(nodes),
            "relation_count": len(links),
            "summary": community_summary
        }
        
        return {
            "nodes": nodes,
            "links": links,
            "community_info": stats
        }
            
    except Exception as e:
        print(f"获取社区信息失败: {str(e)}")
        return None

@router.get("/relation_types")
async def get_relation_type_list():
    """获取图谱中所有可能的关系类型"""
    try:
        db_manager = get_db_manager()
        driver = db_manager.get_driver()
        types = get_relation_types(driver)
        return {"relation_types": types}
    except Exception as e:
        return {"error": str(e), "relation_types": []}