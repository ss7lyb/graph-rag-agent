from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
from neo4j import Result
import traceback
import threading
import time
import functools
import re

import shutup
shutup.please()
import jieba
import jieba.analyse

from langchain_core.messages import RemoveMessage, AIMessage, HumanMessage, ToolMessage
from agent.graph_agent import GraphAgent
from agent.hybrid_agent import HybridAgent
from agent.naive_rag_agent import NaiveRagAgent
from config.neo4jdb import get_db_manager

# 初始化 FastAPI 应用
app = FastAPI()

# 初始化 Neo4j 数据库连接
db_manager = get_db_manager()
driver = db_manager.driver

# 创建线程锁和时间戳字典用于并发控制
feedback_locks = {}
operation_timestamps = {}
chat_locks = {}
chat_timestamps = {}

# 创建一个字典来存储所有可用的 Agent
agents = {
    "graph_agent": GraphAgent(),
    "hybrid_agent": HybridAgent(),
    "naive_rag_agent": NaiveRagAgent(),
}

# ================ 数据模型定义 ================

class ChatRequest(BaseModel):
    message: str
    session_id: str
    debug: bool = False
    agent_type: str = "graph_agent"  # 默认采用 graphrag

class ChatResponse(BaseModel):
    answer: str
    execution_log: Optional[List[Dict]] = None
    kg_data: Optional[Dict] = None

class SourceRequest(BaseModel):
    source_id: str

class SourceResponse(BaseModel):
    content: str

class ClearRequest(BaseModel):
    session_id: str

class ClearResponse(BaseModel):
    status: str
    remaining_messages: Optional[str] = None

class FeedbackRequest(BaseModel):
    message_id: str
    query: str
    is_positive: bool
    thread_id: str
    agent_type: Optional[str] = "graph_agent"

class FeedbackResponse(BaseModel):
    status: str
    action: str

# ================ 工具函数 ================

def measure_performance(endpoint_name):
    """性能测量装饰器"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # 记录性能
                duration = time.time() - start_time
                print(f"API性能 - {endpoint_name}: {duration:.4f}s")
                
                return result
            except Exception as e:
                # 记录异常和性能
                duration = time.time() - start_time
                print(f"API异常 - {endpoint_name}: {str(e)} ({duration:.4f}s)")
                raise
                
        return wrapper
    return decorator

def format_messages_for_response(messages: List[Dict]) -> str:
    """将消息格式化为字符串"""
    formatted = []
    for msg in messages:
        if isinstance(msg, (HumanMessage, AIMessage)):
            prefix = "User: " if isinstance(msg, HumanMessage) else "AI: "
            formatted.append(f"{prefix}{msg.content}")
    return "\\n".join(formatted)

def format_execution_log(log: List[Dict]) -> List[Dict]:
    """格式化执行日志用于JSON响应"""
    formatted_log = []
    for entry in log:
        if isinstance(entry["input"], dict):
            input_str = {}
            for k, v in entry["input"].items():
                if isinstance(v, str):
                    input_str[k] = v
                else:
                    input_str[k] = str(v)
        else:
            input_str = str(entry["input"])
            
        if isinstance(entry["output"], dict):
            output_str = {}
            for k, v in entry["output"].items():
                if isinstance(v, str):
                    output_str[k] = v
                else:
                    output_str[k] = str(v)
        else:
            output_str = str(entry["output"])

        formatted_entry = {
            "node": entry["node"],
            "input": input_str,
            "output": output_str
        }
        formatted_log.append(formatted_entry)
    return formatted_log

def extract_smart_keywords(query):
    """使用jieba智能提取中文关键词"""
    if not query:
        return []
        
    try:        
        # 常见的停用词
        stop_words = ['什么', '之间', '发生', '关系', '的', '和', '有', '是', '在', '了', '吗',
                     '为什么', '如何', '怎么', '怎样', '请问', '告诉', '我', '你', '他', '她', '它',
                     '们', '这个', '那个', '这些', '那些', '一个', '一些', '一下', '地', '得', '着']
        
        # 使用TF-IDF提取关键词
        tfidf_keywords = jieba.analyse.extract_tags(query, topK=3)
        
        # 使用TextRank提取关键词
        textrank_keywords = jieba.analyse.textrank(query, topK=3)
        
        # 使用精确模式分词提取2个字以上的词
        seg_list = jieba.cut(query, cut_all=False)
        seg_words = [word for word in seg_list if len(word) >= 2 and word not in stop_words]
        
        # 合并关键词并去重
        all_keywords = list(set(tfidf_keywords + textrank_keywords + seg_words))
        
        # 按长度排序，优先使用长词
        all_keywords.sort(key=len, reverse=True)
        
        # 如果关键词超过5个，只取前5个
        result = all_keywords[:5] if len(all_keywords) > 5 else all_keywords
        
        # 如果没有提取到关键词，尝试直接提取实体名称
        if not result:
            # 匹配实体名称
            import re
            entity_names = re.findall(r'[\u4e00-\u9fa5]{2,}', query)
            result = [name for name in entity_names if name not in stop_words]
        
        return result
        
    except ImportError:
        print("jieba库未安装，使用简单分词")
        # 回退到简单的正则匹配
        import re
        words = re.findall(r'[\u4e00-\u9fa5]{2,}|[a-zA-Z]{2,}', query)
        stop_words = ['什么', '之间', '发生', '关系', '的', '和', '有', '是', '在', '了', '吗']
        return [w for w in words if w not in stop_words]
    except Exception as e:
        print(f"关键词提取失败: {e}")
        # 最后回退到直接分割
        return [query]

# ================ 知识图谱相关函数 ================

def extract_kg_from_message(message: str, query: str = None) -> Dict:
    """从消息中提取知识图谱实体和关系数据"""
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

def check_entity_existence(entity_ids):
    """检查实体ID是否存在于数据库中"""
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

def inspect_entity_by_id(entity_id):
    """检查特定ID的实体详情"""
    try:
        query = """
        MATCH (e) 
        WHERE e.id = $id
        RETURN e, labels(e) as labels
        """
        
        result = driver.execute_query(query, {"id": entity_id})
        
        if result.records and len(result.records) > 0:
            return True
        else:
            print(f"未找到ID为 {entity_id} 的实体")
            return False
            
    except Exception as e:
        print(f"检查实体详情时出错: {str(e)}")
        return False

def get_entities_from_chunk(chunk_id):
    """根据文本块ID查询相关联的实体"""
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

def get_graph_from_chunks(chunk_ids):
    """直接从文本块获取知识图谱"""
    try:
        print(f"从文本块获取知识图谱: {chunk_ids}")
        
        query = """
        // 通过文本块直接查询相关实体
        MATCH (c:__Chunk__)-[:MENTIONS]->(e:__Entity__)
        WHERE c.id IN $chunk_ids
        
        // 获取这些实体之间的关系
        WITH collect(DISTINCT e) AS entities
        UNWIND entities AS e1
        OPTIONAL MATCH (e1)-[r]-(e2:__Entity__)
        WHERE e2 IN entities
        
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
        [r IN collect(r) WHERE r IS NOT NULL | {
            source: startNode(r).id,
            target: endNode(r).id,
            label: type(r),
            weight: 1
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

def get_knowledge_graph_for_ids(entity_ids=None, relationship_ids=None, chunk_ids=None):
    """根据ID获取知识图谱数据"""
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
        
        # 简化的Cypher查询，专注于可靠地查询实体
        query = """
        // 匹配指定的实体ID
        MATCH (e:__Entity__)
        WHERE e.id IN $entity_ids
        
        // 收集基础实体
        WITH collect(e) AS base_entities
        
        // 展开基础实体以便后续处理
        UNWIND base_entities AS base_entity
        
        // 获取一跳邻居
        OPTIONAL MATCH (base_entity)-[r]-(neighbor:__Entity__)
        WHERE neighbor <> base_entity
        
        // 收集所有实体和关系
        WITH base_entities, 
             collect(DISTINCT neighbor) AS neighbors,
             collect(DISTINCT r) AS all_rels
        
        // 合并所有实体
        WITH base_entities + neighbors AS all_entities, all_rels
        
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
        [r IN all_rels | {
            source: startNode(r).id, 
            target: endNode(r).id, 
            label: type(r),
            weight: CASE WHEN r.weight IS NULL THEN 1 ELSE r.weight END
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

# ================ API路由 ================

@app.post("/chat", response_model=ChatResponse)
@measure_performance("chat")
async def chat(request: ChatRequest):
    """处理聊天请求"""
    # 生成锁的键
    lock_key = f"{request.session_id}_chat"
    
    # 确保每个会话ID有自己的锁
    if lock_key not in chat_locks:
        chat_locks[lock_key] = threading.Lock()
        chat_timestamps[lock_key] = time.time()
    
    # 非阻塞方式尝试获取锁
    lock_acquired = chat_locks[lock_key].acquire(blocking=False)
    if not lock_acquired:
        # 如果无法获取锁，说明有另一个请求正在处理
        raise HTTPException(
            status_code=429, 
            detail="当前有其他请求正在处理，请稍后再试"
        )
    
    try:
        # 更新操作时间戳
        chat_timestamps[lock_key] = time.time()
        
        # 检查请求的agent_type是否存在
        if request.agent_type not in agents:
            raise HTTPException(status_code=400, detail=f"未知的agent类型: {request.agent_type}")
            
        # 获取指定的agent
        selected_agent = agents[request.agent_type]
        
        # 首先尝试快速路径 - 跳过完整处理
        try:
            start_fast = time.time()
            fast_result = selected_agent.check_fast_cache(request.message, request.session_id)
            
            if fast_result:
                print(f"API快速路径命中: {time.time() - start_fast:.4f}s")
                
                # 在调试模式下，需要提供额外信息
                if request.debug:
                    # 提供模拟的执行日志
                    mock_log = [{
                        "node": "fast_cache_hit", 
                        "timestamp": time.time(), 
                        "input": request.message, 
                        "output": "高质量缓存命中，跳过完整处理"
                    }]
                    
                    # 尝试提取图谱数据
                    try:
                        kg_data = extract_kg_from_message(fast_result)
                    except:
                        kg_data = {"nodes": [], "links": []}
                        
                    return ChatResponse(
                        answer=fast_result,
                        execution_log=format_execution_log(mock_log),
                        kg_data=kg_data
                    )
                else:
                    # 标准模式直接返回答案
                    return ChatResponse(answer=fast_result)
        except Exception as e:
            # 快速路径失败，继续常规流程
            print(f"快速路径检查失败: {e}")
        
        if request.debug:
            # 在Debug模式下使用ask_with_trace，并返回知识图谱数据
            result = selected_agent.ask_with_trace(
                request.message, 
                thread_id=request.session_id
            )
            
            # 从结果中提取知识图谱数据
            kg_data = extract_kg_from_message(result["answer"])
            
            return ChatResponse(
                answer=result["answer"],
                execution_log=format_execution_log(result["execution_log"]),
                kg_data=kg_data
            )
        else:
            # 标准模式
            answer = selected_agent.ask(
                request.message, 
                thread_id=request.session_id
            )
            return ChatResponse(answer=answer)
    except Exception as e:
        print(f"处理聊天请求时出错: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 释放锁
        chat_locks[lock_key].release()
        
        # 清理过期的锁
        current_time = time.time()
        expired_keys = []
        for k in chat_timestamps:
            if current_time - chat_timestamps[k] > 300:  # 5分钟后清除
                expired_keys.append(k)
        
        for k in expired_keys:
            if k in chat_locks:
                # 确保锁是可以被安全删除的
                try:
                    if not chat_locks[k].locked():
                        del chat_locks[k]
                except:
                    pass
            if k in chat_timestamps:
                del chat_timestamps[k]

@app.post("/clear", response_model=ClearResponse)
async def clear_chat(request: ClearRequest):
    """清除聊天历史"""
    try:
        # 清除所有agent的历史
        for agent_name, agent in agents.items():
            config = {"configurable": {"thread_id": request.session_id}}
            
            # 添加检查，防止None值报错
            memory_content = agent.memory.get(config)
            if memory_content is None or "channel_values" not in memory_content:
                continue  # 跳过这个agent
                
            messages = memory_content["channel_values"]["messages"]
            
            # 如果消息少于2条，不进行删除操作
            if len(messages) <= 2:
                continue

            i = len(messages)
            for message in reversed(messages):
                if isinstance(messages[2], ToolMessage) and i == 4:
                    break
                agent.graph.update_state(config, {"messages": RemoveMessage(id=message.id)})
                i = i - 1
                if i == 2:  # 保留前两条消息
                    break

        # 获取剩余消息
        try:
            # 使用graph_agent检查剩余消息
            graph_agent = agents["graph_agent"]
            memory_content = graph_agent.memory.get({"configurable": {"thread_id": request.session_id}})
            remaining_text = ""
            
            if memory_content and "channel_values" in memory_content:
                remaining_messages = memory_content["channel_values"]["messages"]
                for msg in remaining_messages:
                    if isinstance(msg, (AIMessage, HumanMessage)):
                        prefix = "AI: " if isinstance(msg, AIMessage) else "User: "
                        remaining_text += f"{prefix}{msg.content}\n"
        except:
            remaining_text = ""
        
        return ClearResponse(
            status="success",
            remaining_messages=remaining_text
        )
            
    except Exception as e:
        print(f"清除聊天历史时出错: {str(e)}")
        return ClearResponse(
            status="success",
            remaining_messages=""
        )

@app.post("/source", response_model=SourceResponse)
async def get_source(request: SourceRequest):
    """处理源内容请求"""
    try:
        source_id = request.source_id
        
        if not source_id:
            return SourceResponse(content="未提供有效的源ID")
        
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
            
        return SourceResponse(content=content)
    except Exception as e:
        print(f"获取源内容时出错: {str(e)}")
        return SourceResponse(content=f"检索源内容时发生错误: {str(e)}")

@app.get("/knowledge_graph")
async def get_knowledge_graph(limit: int = 100, query: str = None):
    """获取知识图谱数据"""
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

@app.get("/knowledge_graph_from_message")
async def get_knowledge_graph_from_message(message: str = None, query: str = None):
    """从消息文本中提取知识图谱数据"""
    if not message:
        return {"nodes": [], "links": []}
        
    try:
        return extract_kg_from_message(message, query)
            
    except Exception as e:
        print(f"从消息提取知识图谱失败: {str(e)}")
        return {"error": str(e), "nodes": [], "links": []}

@app.get("/entity_types")
async def get_entity_types():
    """获取数据库中所有可能的实体类型"""
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
            
        return {"entity_types": entity_types}
        
    except Exception as e:
        print(f"获取实体类型失败: {str(e)}")
        return {"error": str(e), "entity_types": []}

@app.get("/chunks")
async def get_chunks(limit: int = 10, offset: int = 0):
    """获取数据库中的文本块"""
    try:
        query = """
        MATCH (c:__Chunk__)
        RETURN c.id AS id, c.fileName AS fileName, c.text AS text
        ORDER BY c.fileName, c.id
        SKIP $offset
        LIMIT $limit
        """
        
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

@app.post("/feedback", response_model=FeedbackResponse)
@measure_performance("feedback")
async def process_feedback(request: FeedbackRequest):
    """处理用户对回答的反馈"""
    try:
        # 生成锁的键
        lock_key = f"{request.thread_id}_{request.query}"
        
        # 确保每个线程ID+查询有自己的锁
        if lock_key not in feedback_locks:
            feedback_locks[lock_key] = threading.Lock()
            operation_timestamps[lock_key] = time.time()
        
        # 获取锁，防止并发处理同一个查询
        with feedback_locks[lock_key]:
            # 动态获取对应的agent
            agent_type = request.agent_type if hasattr(request, 'agent_type') else "graph_agent"
            
            # 确保agent_type存在
            if not agent_type or agent_type not in agents:
                agent_type = "graph_agent"  # 回退到默认agent
                print(f"未知的agent类型，使用默认值: {agent_type}")
                
            selected_agent = agents[agent_type]
            
            # 根据反馈进行处理
            if request.is_positive:
                # 标记为高质量回答
                selected_agent.mark_answer_quality(request.query, True, request.thread_id)
                action = "缓存已被标记为高质量"
            else:
                # 负面反馈 - 从缓存中移除该回答
                selected_agent.clear_cache_for_query(request.query, request.thread_id)
                action = "缓存已被清除"
                
            # 更新操作时间戳
            operation_timestamps[lock_key] = time.time()
                
            # 清理过期的锁
            current_time = time.time()
            expired_keys = []
            for k in operation_timestamps:
                if current_time - operation_timestamps[k] > 300:  # 5分钟后清除
                    expired_keys.append(k)
            
            for k in expired_keys:
                if k in feedback_locks:
                    del feedback_locks[k]
                if k in operation_timestamps:
                    del operation_timestamps[k]
            
            return FeedbackResponse(
                status="success",
                action=action
            )
    except Exception as e:
        print(f"处理反馈时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
def shutdown_event():
    """应用关闭时清理资源"""
    for agent_name, agent in agents.items():
        try:
            agent.close()
            print(f"已关闭 {agent_name} 资源")
        except Exception as e:
            print(f"关闭 {agent_name} 资源时出错: {e}")
    
    # 关闭Neo4j连接
    if driver:
        driver.close()
        print("已关闭Neo4j连接")

# 启动服务器
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)