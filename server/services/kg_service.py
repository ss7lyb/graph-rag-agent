import re
import traceback
from typing import Dict, List, Any
from server_config.database import get_db_manager
from utils.keywords import extract_smart_keywords


# 获取数据库连接
db_manager = get_db_manager()
driver = db_manager.driver


def extract_kg_from_message(message: str, query: str = None) -> Dict:
    """
    从消息中提取知识图谱实体和关系数据
    
    Args:
        message: 消息文本
        query: 用户查询内容(可选)
    
    Returns:
        Dict: 知识图谱数据，包含节点和连接
    """
    try:
        # 直接使用正则表达式提取各部分数据
        entity_ids = []
        rel_ids = []
        chunk_ids = []
        
        # 匹配 Entities 列表
        entity_pattern = r"['\"]?Entities['\"]?\s*:\s*\[(.*?)\]"
        entity_match = re.search(entity_pattern, message, re.DOTALL)
        if entity_match:
            entity_str = entity_match.group(1).strip()
            try:
                # 处理数字ID
                entity_parts = [p.strip() for p in entity_str.split(',') if p.strip()]
                for part in entity_parts:
                    clean_part = part.strip("'\"")
                    if clean_part.isdigit():
                        entity_ids.append(int(clean_part))
                    else:
                        entity_ids.append(clean_part)
            except Exception as e:
                print(f"解析实体ID时出错: {e}")
        
        # 匹配 Relationships 或 Reports 列表
        rel_pattern = r"['\"]?(?:Relationships|Reports)['\"]?\s*:\s*\[(.*?)\]"
        rel_match = re.search(rel_pattern, message, re.DOTALL)
        if rel_match:
            rel_str = rel_match.group(1).strip()
            try:
                # 处理数字ID
                rel_parts = [p.strip() for p in rel_str.split(',') if p.strip()]
                for part in rel_parts:
                    clean_part = part.strip("'\"")
                    if clean_part.isdigit():
                        rel_ids.append(int(clean_part))
                    else:
                        rel_ids.append(clean_part)
            except Exception as e:
                print(f"解析关系ID时出错: {e}")
        
        # 匹配 Chunks 列表
        chunk_pattern = r"['\"]?Chunks['\"]?\s*:\s*\[(.*?)\]"
        chunk_match = re.search(chunk_pattern, message, re.DOTALL)
        if chunk_match:
            chunks_str = chunk_match.group(1).strip()
            
            # 处理带引号的chunk IDs
            if "'" in chunks_str or '"' in chunks_str:
                # 匹配所有被引号包围的内容
                chunk_parts = re.findall(r"['\"]([^'\"]*)['\"]", chunks_str)
                chunk_ids = [part for part in chunk_parts if part]
            else:
                # 没有引号的情况，直接分割
                chunk_ids = [part.strip() for part in chunks_str.split(',') if part.strip()]
        
        # 提取关键词 (可选)
        query_keywords = []
        if query:
            query_keywords = extract_smart_keywords(query)
        
        # 获取知识图谱
        return get_knowledge_graph_for_ids(entity_ids, rel_ids, chunk_ids)
        
    except Exception as e:
        print(f"提取知识图谱数据失败: {str(e)}")
        traceback.print_exc()
        return {"nodes": [], "links": []}


def check_entity_existence(entity_ids: List[Any]) -> List:
    """
    检查实体ID是否存在于数据库中
    
    Args:
        entity_ids: 实体ID列表
    
    Returns:
        List: 确认存在的实体ID列表
    """
    try:
        # 尝试多种格式查询，确保能找到实体
        query = """
        // 尝试不同格式匹配实体ID
        UNWIND $ids AS id
        OPTIONAL MATCH (e:__Entity__) 
        WHERE e.id = id OR 
              e.id = toString(id) OR
              toString(e.id) = toString(id)
        RETURN id AS input_id, e.id AS found_id, labels(e) AS labels
        """
        
        params = {"ids": entity_ids}
        
        result = driver.execute_query(query, params)
        
        if result.records:
            found_entities = [r.get("found_id") for r in result.records if r.get("found_id") is not None]
            return found_entities
        else:
            print("没有找到任何匹配的实体")
            return []
            
    except Exception as e:
        print(f"检查实体ID时出错: {str(e)}")
        return []


def get_entities_from_chunk(chunk_id: str) -> List:
    """
    根据文本块ID查询相关联的实体
    
    Args:
        chunk_id: 文本块ID
    
    Returns:
        List: 与该文本块关联的实体ID列表
    """
    try:
        query = """
        MATCH (c:__Chunk__)-[:MENTIONS]->(e:__Entity__)
        WHERE c.id = $chunk_id
        RETURN collect(distinct e.id) AS entity_ids
        """
        
        params = {"chunk_id": chunk_id}
        
        result = driver.execute_query(query, params)
        
        if result.records and len(result.records) > 0:
            entity_ids = result.records[0].get("entity_ids", [])
            return entity_ids
        else:
            print(f"文本块 {chunk_id} 没有关联的实体")
            return []
            
    except Exception as e:
        print(f"查询文本块关联实体时出错: {str(e)}")
        return []


def get_graph_from_chunks(chunk_ids: List[str]) -> Dict:
    """
    直接从文本块获取知识图谱
    
    Args:
        chunk_ids: 文本块ID列表
    
    Returns:
        Dict: 知识图谱数据，包含节点和连接
    """
    try:
        print(f"从文本块获取知识图谱: {chunk_ids}")
        
        query = """
        // 通过文本块直接查询相关实体
        MATCH (c:__Chunk__)-[:MENTIONS]->(e:__Entity__)
        WHERE c.id IN $chunk_ids
        
        // 获取这些实体集合
        WITH collect(DISTINCT e) AS entities
        
        // 处理实体间的关系 - 只处理每对实体一次
        UNWIND entities AS e1
        UNWIND entities AS e2
        // 确保只处理每对实体一次
        WITH entities, e1, e2 
        WHERE e1.id < e2.id
        OPTIONAL MATCH (e1)-[r]-(e2)
        
        // 收集关系
        WITH entities, e1, e2, collect(r) AS rels
        
        // 构建去重的关系集合
        WITH entities, 
             collect({
                 source: e1.id, 
                 target: e2.id, 
                 rels: rels
             }) AS relations
        
        // 扁平化和去重关系
        WITH entities,
             [rel IN relations WHERE size(rel.rels) > 0 |
              // 为每种类型的关系创建唯一记录
              [r IN rel.rels | {
                source: rel.source,
                target: rel.target,
                relType: type(r),
                label: type(r),
                weight: 1
              }]
             ] AS links_nested
             
        // 扁平化嵌套关系
        WITH entities,
             REDUCE(acc = [], list IN links_nested | acc + list) AS all_links
        
        // 最终去重，基于源、目标和关系类型
        WITH entities,
             [link IN all_links | 
              link.source + '_' + link.target + '_' + link.relType
             ] AS link_keys,
             all_links
        
        // 只保留唯一的关系
        WITH entities,
             [i IN RANGE(0, size(all_links)-1) WHERE 
              i = REDUCE(min_i = i, j IN RANGE(0, size(all_links)-1) |
                   CASE WHEN link_keys[j] = link_keys[i] AND j < min_i
                        THEN j ELSE min_i END)
             | all_links[i]
             ] AS unique_links
             
        // 收集结果
        RETURN 
        [e IN entities | {
            id: e.id,
            label: e.id,
            description: CASE WHEN e.description IS NULL THEN '' ELSE e.description END,
            group: CASE 
                WHEN [lbl IN labels(e) WHERE lbl <> '__Entity__'] <> []
                THEN [lbl IN labels(e) WHERE lbl <> '__Entity__'][0]
                ELSE 'Unknown'
            END
        }] AS nodes,
        [link IN unique_links | {
            source: link.source,
            target: link.target,
            label: link.label,
            weight: link.weight
        }] AS links
        """
        
        result = driver.execute_query(query, {"chunk_ids": chunk_ids})
        
        if not result.records or len(result.records) == 0:
            print("从文本块查询结果为空")
            return {"nodes": [], "links": []}
            
        record = result.records[0]
        nodes = record.get("nodes", [])
        links = record.get("links", [])
        print(f"从文本块查询结果: {len(nodes)} 个节点, {len(links)} 个连接")
        
        return {
            "nodes": nodes,
            "links": links
        }
        
    except Exception as e:
        print(f"从文本块获取知识图谱失败: {str(e)}")
        return {"nodes": [], "links": []}


def get_knowledge_graph_for_ids(entity_ids=None, relationship_ids=None, chunk_ids=None) -> Dict:
    """
    根据ID获取知识图谱数据
    
    Args:
        entity_ids: 实体ID列表(可选)
        relationship_ids: 关系ID列表(可选)
        chunk_ids: 文本块ID列表(可选)
    
    Returns:
        Dict: 知识图谱数据，包含节点和连接
    """
    try:
        # 确保所有参数都有默认值，避免None
        entity_ids = entity_ids or []
        relationship_ids = relationship_ids or []
        chunk_ids = chunk_ids or []
        
        # 如果提供了文本块ID，但没有实体ID，尝试从文本块获取实体
        if chunk_ids and not entity_ids:
            for chunk_id in chunk_ids:
                chunk_entities = get_entities_from_chunk(chunk_id)
                entity_ids.extend(chunk_entities)
            
            # 去重
            entity_ids = list(set(entity_ids))
        
        if not entity_ids and not chunk_ids:
            return {"nodes": [], "links": []}
        
        # 检查实体ID是否存在
        verified_entity_ids = check_entity_existence(entity_ids)
        if not verified_entity_ids:
            # 尝试直接使用文本块查询
            if chunk_ids:
                return get_graph_from_chunks(chunk_ids)
            return {"nodes": [], "links": []}
        
        # 使用确认存在的实体ID进行查询
        params = {
            "entity_ids": verified_entity_ids,
            "max_distance": 1
        }
        
        # 局部查询的Cypher
        query = """
        // 匹配指定的实体ID
        MATCH (e:__Entity__)
        WHERE e.id IN $entity_ids
        
        // 收集基础实体
        WITH collect(e) AS base_entities
        
        // 匹配实体之间的关系，只处理每对实体一次
        UNWIND base_entities AS e1
        UNWIND base_entities AS e2
        // 确保只处理每对实体一次
        WITH base_entities, e1, e2 
        WHERE e1.id < e2.id
        OPTIONAL MATCH (e1)-[r]-(e2)
        
        // 收集关系
        WITH base_entities, e1, e2, collect(r) AS rels
        
        // 获取一跳邻居，排除已经处理过的实体对
        UNWIND base_entities AS base_entity
        OPTIONAL MATCH (base_entity)-[r1]-(neighbor:__Entity__)
        WHERE NOT neighbor IN base_entities
        
        // 收集所有实体和关系
        WITH base_entities, 
             collect(DISTINCT {source: e1.id, target: e2.id, rels: rels}) AS internal_rels,
             collect(DISTINCT neighbor) AS neighbors,
             collect(DISTINCT {source: base_entity.id, target: neighbor.id, rel: r1}) AS external_rels
        
        // 合并所有实体
        WITH base_entities + neighbors AS all_entities, 
             internal_rels, external_rels
        
        // 构建去重的内部关系
        WITH all_entities,
             [rel IN internal_rels WHERE size(rel.rels) > 0 |
              // 为每种类型的关系创建一个唯一记录
              [r IN rel.rels | {
                source: rel.source,
                target: rel.target,
                label: type(r),
                relType: type(r),
                weight: CASE WHEN r.weight IS NULL THEN 1 ELSE r.weight END
              }]
             ] AS internal_links_nested,
             
             // 构建去重的外部关系
             [rel IN external_rels WHERE rel.rel IS NOT NULL |
              {
                source: rel.source,
                target: rel.target,
                label: type(rel.rel),
                relType: type(rel.rel),
                weight: CASE WHEN rel.rel.weight IS NULL THEN 1 ELSE rel.rel.weight END
              }
             ] AS external_links
        
        // 扁平化内部关系并合并
        WITH all_entities,
             [link IN external_links | link] + 
             [link IN REDUCE(acc = [], list IN internal_links_nested | acc + list) | link]
             AS all_links_raw
        
        // 最终去重，基于源、目标和关系类型
        WITH all_entities,
             [link IN all_links_raw | 
              link.source + '_' + link.target + '_' + link.relType
             ] AS link_keys,
             all_links_raw
        
        // 只保留唯一的关系
        WITH all_entities,
             [i IN RANGE(0, size(all_links_raw)-1) WHERE 
              i = REDUCE(min_i = i, j IN RANGE(0, size(all_links_raw)-1) |
                   CASE WHEN link_keys[j] = link_keys[i] AND j < min_i
                        THEN j ELSE min_i END)
             | all_links_raw[i]
             ] AS unique_links
        
        // 返回结果
        RETURN 
        [n IN all_entities | {
            id: n.id, 
            label: CASE WHEN n.id IS NULL THEN "未知" ELSE n.id END, 
            description: CASE WHEN n.description IS NULL THEN '' ELSE n.description END,
            group: CASE 
                WHEN [lbl IN labels(n) WHERE lbl <> '__Entity__'] <> []
                THEN [lbl IN labels(n) WHERE lbl <> '__Entity__'][0]
                ELSE 'Unknown'
            END
        }] AS nodes,
        [link IN unique_links | {
            source: link.source,
            target: link.target,
            label: link.label,
            weight: link.weight
        }] AS links
        """
        
        # 执行查询
        result = driver.execute_query(query, params)
        
        if not result.records or len(result.records) == 0:
            # 尝试直接使用文本块查询
            if chunk_ids:
                return get_graph_from_chunks(chunk_ids)
            return {"nodes": [], "links": []}
            
        record = result.records[0]
        nodes = record.get("nodes", [])
        links = record.get("links", [])
        
        return {
            "nodes": nodes,
            "links": links
        }
        
    except Exception as e:
        print(f"获取知识图谱失败: {str(e)}")
        
        # 尝试直接使用文本块查询
        if chunk_ids:
            return get_graph_from_chunks(chunk_ids)
        return {"nodes": [], "links": []}


def get_knowledge_graph(limit: int = 100, query: str = None) -> Dict:
    """
    获取知识图谱数据
    
    Args:
        limit: 节点数量限制
        query: 查询条件(可选)
    
    Returns:
        Dict: 知识图谱数据，包含节点和连接
    """
    try:
        # 确保limit是整数
        limit = int(limit) if limit else 100
        
        # 构建查询条件
        query_conditions = ""
        params = {"limit": limit}
        
        if query:
            query_conditions = """
            WHERE n.id CONTAINS $query OR 
                  n.description CONTAINS $query
            """
            params["query"] = query
        else:
            query_conditions = ""
            
        # 构建节点查询 - 动态获取节点类型
        node_query = f"""
        // 获取实体
        MATCH (n:__Entity__)
        {query_conditions}
        WITH n LIMIT $limit
        
        // 收集所有实体
        WITH collect(n) AS entities
        
        // 获取实体间的关系
        CALL {{
            WITH entities
            MATCH (e1:__Entity__)-[r]-(e2:__Entity__)
            WHERE e1 IN entities AND e2 IN entities
                AND e1.id < e2.id  // 避免重复关系
            RETURN collect(r) AS relationships
        }}
        
        // 返回结果
        RETURN 
        [entity IN entities | {{
            id: entity.id,
            label: entity.id,
            description: entity.description,
            // 动态使用实体标签作为组
            group: CASE 
                WHEN [lbl IN labels(entity) WHERE lbl <> '__Entity__'] <> []
                THEN [lbl IN labels(entity) WHERE lbl <> '__Entity__'][0]
                ELSE 'Unknown'
            END
        }}] AS nodes,
        [r IN relationships | {{
            source: startNode(r).id,
            target: endNode(r).id,
            label: type(r),
            weight: CASE WHEN r.weight IS NOT NULL THEN r.weight ELSE 1 END
        }}] AS links
        """
        
        result = driver.execute_query(node_query, params)
        
        if not result or not result.records:
            return {"nodes": [], "links": []}
            
        record = result.records[0]
        
        # 处理可能的None值
        nodes = record["nodes"] or []
        links = record["links"] or []
        
        # 返回标准格式
        return {
            "nodes": nodes,
            "links": links
        }
        
    except Exception as e:
        print(f"获取知识图谱数据失败: {str(e)}")
        return {"error": str(e), "nodes": [], "links": []}


def get_entity_types() -> List[str]:
    """
    获取数据库中所有可能的实体类型
    
    Returns:
        List[str]: 实体类型列表
    """
    try:
        query = """
        MATCH (n:__Entity__)
        UNWIND labels(n) AS label
        WHERE label <> '__Entity__'
        RETURN DISTINCT label
        ORDER BY label
        """
        
        result = driver.execute_query(query)
        
        entity_types = []
        for record in result.records:
            entity_types.append(record["label"])
            
        return entity_types
        
    except Exception as e:
        print(f"获取实体类型失败: {str(e)}")
        return []


def get_source_content(source_id: str) -> str:
    """
    根据源ID获取内容
    
    Args:
        source_id: 源ID
        
    Returns:
        str: 源内容
    """
    try:
        if not source_id:
            return "未提供有效的源ID"
        
        # 检查ID是否为Chunk ID (直接使用)
        if len(source_id) == 40:  # SHA1哈希的长度
            query = """
            MATCH (n:__Chunk__) 
            WHERE n.id = $id 
            RETURN n.fileName AS fileName, n.text AS text
            """
            params = {"id": source_id}
        else:
            # 尝试解析复合ID
            id_parts = source_id.split(",")
            
            if len(id_parts) >= 2 and id_parts[0] == "2":  # 文本块查询
                query = """
                MATCH (n:__Chunk__) 
                WHERE n.id = $id 
                RETURN n.fileName AS fileName, n.text AS text
                """
                params = {"id": id_parts[-1]}
            else:  # 社区查询
                query = """
                MATCH (n:__Community__) 
                WHERE n.id = $id 
                RETURN n.summary AS summary, n.full_content AS full_content
                """
                params = {"id": id_parts[1] if len(id_parts) > 1 else source_id}
        
        from neo4j import Result
        result = driver.execute_query(
            query,
            params,
            result_transformer_=Result.to_df
        )
        
        if result is not None and result.shape[0] > 0:
            if "text" in result.columns:
                content = f"文件名: {result.iloc[0]['fileName']}\n\n{result.iloc[0]['text']}"
            else:
                content = f"摘要:\n{result.iloc[0]['summary']}\n\n全文:\n{result.iloc[0]['full_content']}"
        else:
            content = f"未找到相关内容: 源ID {source_id}"
            
        return content
    except Exception as e:
        print(f"获取源内容时出错: {str(e)}")
        return f"检索源内容时发生错误: {str(e)}"


def get_chunks(limit: int = 10, offset: int = 0):
    """
    获取数据库中的文本块
    
    Args:
        limit: 返回数量限制
        offset: 偏移量
        
    Returns:
        Dict: 文本块数据和总数
    """
    try:
        query = """
        MATCH (c:__Chunk__)
        RETURN c.id AS id, c.fileName AS fileName, c.text AS text
        ORDER BY c.fileName, c.id
        SKIP $offset
        LIMIT $limit
        """
        
        from neo4j import Result
        result = driver.execute_query(
            query, 
            parameters={"limit": int(limit), "offset": int(offset)},
            result_transformer_=Result.to_df
        )
        
        if result is not None and not result.empty:
            chunks = result.to_dict(orient='records')
            return {"chunks": chunks, "total": len(chunks)}
        else:
            return {"chunks": [], "total": 0}
            
    except Exception as e:
        print(f"获取文本块失败: {str(e)}")
        return {"error": str(e), "chunks": []}