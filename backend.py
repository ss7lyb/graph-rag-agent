import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
from neo4j import GraphDatabase, Result
from dotenv import load_dotenv
import traceback
import re
import threading
import time
import functools

load_dotenv()

import shutup
shutup.please()

from langchain_core.messages import RemoveMessage, AIMessage, HumanMessage, ToolMessage
from agent.graph_agent import GraphAgent
from agent.hybrid_agent import HybridAgent

app = FastAPI()

# 创建线程锁和时间戳字典用于并发控制
feedback_locks = {}
operation_timestamps = {}
chat_locks = {}
chat_timestamps = {}

# 创建一个装饰器来测量API端点的性能
def measure_performance(endpoint_name):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # 记录性能
                duration = time.time() - start_time
                print(f"API性能 - {endpoint_name}: {duration:.4f}s")
                
                # 可以添加到日志或指标系统
                # ...
                
                return result
            except Exception as e:
                # 记录异常和性能
                duration = time.time() - start_time
                print(f"API异常 - {endpoint_name}: {str(e)} ({duration:.4f}s)")
                raise
                
        return wrapper
    return decorator

# 创建一个字典来存储所有可用的Agent
agents = {
    "graph_agent": GraphAgent(),
    "hybrid_agent": HybridAgent(),
}

# 在应用关闭时清理资源
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

# Initialize Neo4j driver
driver = GraphDatabase.driver(
    os.getenv('NEO4J_URI'),
    auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
)

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: str
    debug: bool = False
    agent_type: str = "graph_agent"  # default to graph_agent

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

def format_messages_for_response(messages: List[Dict]) -> str:
    """将消息格式化为字符串"""
    formatted = []
    for msg in messages:
        if isinstance(msg, (HumanMessage, AIMessage)):
            prefix = "User: " if isinstance(msg, HumanMessage) else "AI: "
            formatted.append(f"{prefix}{msg.content}")
    return "\\n".join(formatted)

def format_execution_log(log: List[Dict]) -> List[Dict]:
    """格式化执行日志用于 JSON 响应"""
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

def extract_kg_from_message(message: str) -> Dict:
    """从消息中提取知识图谱实体和关系数据"""
    try:
        # 首先尝试找到引用数据部分 - 通常在"#### 引用数据"之后
        reference_sections = re.findall(r'####\s+引用数据\s*(.*?)(?=###|\Z)', message, re.DOTALL)
        
        entity_ids = []
        rel_ids = []
        chunk_ids = []
        
        if reference_sections:
            # 从引用部分提取数据
            reference_text = reference_sections[0]
            
            # 改进正则表达式，更加灵活地匹配JSON对象
            data_match = re.search(r"\{'data':\s*{(.*?)}\}", reference_text, re.DOTALL) or \
                        re.search(r"{'data':\s*{(.*?)}\}", reference_text, re.DOTALL)
                        
            if data_match:
                data_str = "{" + data_match.group(1) + "}"
                # 更好地清理数据，处理单引号和双引号混用情况
                cleaned_data = data_str.replace("'", '"').replace('"[', '[').replace(']"', ']')
                
                try:
                    # 尝试作为JSON解析
                    import json
                    data_dict = json.loads(cleaned_data)
                    
                    # 提取实体、关系和块ID
                    entity_ids = data_dict.get('Entities', [])
                    rel_ids = data_dict.get('Relationships', []) or data_dict.get('Reports', [])
                    chunk_ids = data_dict.get('Chunks', [])
                    
                    # 确保所有IDs都是字符串列表
                    entity_ids = [str(e) for e in entity_ids]
                    rel_ids = [str(r) for r in rel_ids]
                    
                except json.JSONDecodeError:
                    # 如果JSON解析失败，使用改进的正则表达式提取
                    print("JSON解析失败，尝试使用正则表达式提取数据")
                    
                    # 提取实体IDs - 改进正则匹配模式
                    entities_match = re.search(r"'Entities':\s*\[(.*?)\]", reference_text, re.DOTALL)
                    if entities_match:
                        entity_str = entities_match.group(1).strip()
                        entity_ids = [id.strip() for id in re.findall(r'(\d+|\'[^\']+\')', entity_str) if id.strip()]
                    
                    # 提取关系IDs - 同样改进模式
                    rels_match = re.search(r"'Relationships':\s*\[(.*?)\]", reference_text, re.DOTALL) or \
                                re.search(r"'Reports':\s*\[(.*?)\]", reference_text, re.DOTALL)
                    
                    if rels_match:
                        rel_str = rels_match.group(1).strip()
                        rel_ids = [id.strip() for id in re.findall(r'(\d+|\'[^\']+\')', rel_str) if id.strip()]
                    
                    # 提取Chunk IDs - 使用更精确的模式
                    chunks_match = re.search(r"'Chunks':\s*\[(.*?)\]", reference_text, re.DOTALL)
                    if chunks_match:
                        chunks_str = chunks_match.group(1).strip()
                        # 处理带引号的ID
                        chunk_ids = re.findall(r"'([^']*)'", chunks_str) or re.findall(r'"([^"]*)"', chunks_str)
                        if not chunk_ids:
                            # 处理不带引号的ID
                            chunk_ids = [id.strip() for id in chunks_str.split(',') if id.strip()]
        
        # 打印提取的数据
        print(f"提取的数据: entity_ids={entity_ids}, rel_ids={rel_ids}, chunk_ids={chunk_ids}")
        
        # 查询相关的实体和关系
        if entity_ids or rel_ids or chunk_ids:
            return get_knowledge_graph_for_ids(entity_ids, rel_ids, chunk_ids)
        
        # 如果没有提取到任何有效数据，返回空的知识图谱
        return {"nodes": [], "links": []}
        
    except Exception as e:
        print(f"提取知识图谱数据失败: {str(e)}")
        traceback.print_exc()
        return {"nodes": [], "links": []}

def get_knowledge_graph_for_ids(entity_ids=None, relationship_ids=None, chunk_ids=None, expand_hops=1):
    """根据ID获取知识图谱数据"""
    # 确保所有参数都有默认值，避免None
    entity_ids = entity_ids or []
    relationship_ids = relationship_ids or []
    chunk_ids = chunk_ids or []
    
    if not entity_ids and not relationship_ids and not chunk_ids:
        return {"nodes": [], "links": []}
        
    try:
        # 转换实体ID列表 - 支持数字ID和字符串ID
        numeric_ids = []
        string_ids = []
        
        for eid in entity_ids:
            try:
                if isinstance(eid, str) and eid.isdigit():
                    numeric_ids.append(int(eid))
                elif isinstance(eid, int):
                    numeric_ids.append(eid)
                else:
                    string_ids.append(str(eid))
            except:
                string_ids.append(str(eid))
        
        # 确保查询参数正确设置
        params = {
            "entity_ids": numeric_ids,
            "entity_id_strings": string_ids,
            "chunk_ids": chunk_ids
        }
        
        # 输出参数供调试
        print(f"正在查询特定ID的知识图谱，参数: {params}")
        
        query = """
        // 初始化实体集合
        WITH [] AS entities_set
        
        // 添加从实体ID获取的节点
        CALL {
            // 匹配指定的实体ID
            MATCH (e:__Entity__)
            WHERE ID(e) IN $entity_ids OR e.id IN $entity_id_strings
            RETURN collect(e) AS entity_nodes
        }
        
        // 添加从文本块ID获取的相关实体
        CALL {
            // 匹配指定的文本块，找到关联的实体
            MATCH (c:__Chunk__)-[:MENTIONS]->(e:__Entity__)
            WHERE c.id IN $chunk_ids
            RETURN collect(DISTINCT e) AS chunk_related_entities
        }
        
        // 合并所有实体
        WITH entities_set + entity_nodes + chunk_related_entities AS base_entities
        
        // 扩展周边实体（通过关系连接的）
        CALL {
            WITH base_entities
            MATCH (e)-[r]-(neighbor:__Entity__)
            WHERE e IN base_entities
            RETURN collect(DISTINCT neighbor) AS neighbors
        }
        
        // 合并实体列表
        WITH base_entities + neighbors AS all_entities
        
        // 再次收集所有实体间的关系
        CALL {
            WITH all_entities
            MATCH (e1:__Entity__)-[r]-(e2:__Entity__)
            WHERE e1 IN all_entities AND e2 IN all_entities
            RETURN collect(DISTINCT r) AS all_rels
        }
        
        // 返回结果
        RETURN 
        [n IN all_entities | {
            id: n.id, 
            label: n.id, 
            description: n.description,
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
            weight: CASE WHEN r.weight IS NOT NULL THEN r.weight ELSE 1 END
        }] AS links
        """
        
        # 执行查询
        try:
            result = driver.execute_query(query, params)
            
            if not result.records:
                return {"nodes": [], "links": []}
                
            record = result.records[0]
            return {
                "nodes": record["nodes"],
                "links": record["links"]
            }
            
        except Exception as e:
            print(f"查询知识图谱数据时出错: {str(e)}")
            traceback.print_exc()
            return {"nodes": [], "links": []}
            
    except Exception as e:
        print(f"获取知识图谱失败: {str(e)}")
        traceback.print_exc()
        return {"nodes": [], "links": []}

@app.post("/chat", response_model=ChatResponse)
@measure_performance("chat")
async def chat(request: ChatRequest):
    """处理聊天请求 - 增加并发控制和性能优化"""
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
        
        # 性能优化：首先尝试快速路径 - 跳过完整处理
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

        # 获取剩余消息时也添加检查
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
        traceback.print_exc()
        return ClearResponse(
            status="success",
            remaining_messages=""
        )

@app.post("/source", response_model=SourceResponse)
async def get_source(request: SourceRequest):
    """处理源内容请求"""
    try:
        source_id = request.source_id
        print(f"正在查询源内容，ID: {source_id}")
        
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

        print(f"源内容查询参数: {params}")
        
        result = driver.execute_query(
            query,
            params,
            result_transformer_=Result.to_df
        )
        
        print(f"查询结果: {result.shape if result is not None else 'None'}")
        
        if result is not None and result.shape[0] > 0:
            if "text" in result.columns:
                content = f"文件名: {result.iloc[0]['fileName']}\n\n{result.iloc[0]['text']}"
            else:
                content = f"摘要:\n{result.iloc[0]['summary']}\n\n全文:\n{result.iloc[0]['full_content']}"
        else:
            content = f"未找到相关内容: 源ID {source_id}"
            
        return SourceResponse(content=content)
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"获取源内容时出错: {str(e)}\n{error_trace}")
        return SourceResponse(content=f"检索源内容时发生错误: {str(e)}")

@app.get("/knowledge_graph")
async def get_knowledge_graph(limit: int = 100, query: str = None):
    """获取知识图谱数据，动态检测节点类型"""
    try:
        # 确保limit是整数
        limit = int(limit) if limit else 100
        
        # 构建查询条件
        query_conditions = ""
        params = {"limit": limit}  # 确保传递limit参数
        
        if query:
            query_conditions = """
            WHERE n.id CONTAINS $query OR 
                  n.description CONTAINS $query
            """
            params["query"] = query
        else:
            # 即使没有query参数，也需要确保query_conditions是空字符串而不是None
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
                AND ID(e1) < ID(e2)  // 避免重复关系
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
        
        # 明确输出参数，帮助调试
        print(f"正在查询知识图谱，参数: {params}")
        
        result = driver.execute_query(node_query, params)
        
        if not result or not result.records:
            return {"nodes": [], "links": []}
            
        record = result.records[0]
        
        # 处理可能的None值
        nodes = record["nodes"] or []
        links = record["links"] or []
        
        # 标准化响应格式
        return {
            "nodes": nodes,
            "links": links
        }
        
    except Exception as e:
        print(f"获取知识图谱数据失败: {str(e)}")
        traceback.print_exc()
        return {"error": str(e), "nodes": [], "links": []}

@app.get("/knowledge_graph_from_message")
async def get_knowledge_graph_from_message(message: str = None):
    """
    从消息文本中提取知识图谱数据，使用动态类型
    
    参数:
        message: AI回复消息文本
        
    返回:
        包含节点和链接的知识图谱数据
    """
    if not message:
        return {"nodes": [], "links": []}
        
    try:
        return extract_kg_from_message(message)
            
    except Exception as e:
        print(f"从消息提取知识图谱失败: {str(e)}")
        traceback.print_exc()
        return {"error": str(e), "nodes": [], "links": []}

# 添加获取所有可能的实体类型的API
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
        traceback.print_exc()
        return {"error": str(e), "entity_types": []}

# 添加用于获取文本块内容的API
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
        traceback.print_exc()
        return {"error": str(e), "chunks": []}

@app.post("/feedback", response_model=FeedbackResponse)
@measure_performance("feedback")
async def process_feedback(request: FeedbackRequest):
    """处理用户对回答的反馈 - 增加并发控制和性能优化"""
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
                print(f"未知的agent类型或未提供agent_type，使用默认值: {agent_type}")
                
            selected_agent = agents[agent_type]
            
            # 根据反馈进行处理 - 直接使用优化版方法
            if request.is_positive:
                # 使用优化的反馈标记方法
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
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# 启动服务器
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)