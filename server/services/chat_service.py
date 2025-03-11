import time
import traceback
from typing import Dict
from fastapi import HTTPException

from services.agent_service import agent_manager
from services.kg_service import extract_kg_from_message
from utils.concurrent import chat_manager, feedback_manager


async def process_chat(message: str, session_id: str, debug: bool = False, agent_type: str = "graph_agent") -> Dict:
    """
    处理聊天请求
    
    Args:
        message: 用户消息
        session_id: 会话ID
        debug: 是否为调试模式
        agent_type: Agent类型
        
    Returns:
        Dict: 聊天响应结果
    """
    # 生成锁的键
    lock_key = f"{session_id}_chat"
    
    # 非阻塞方式尝试获取锁
    lock_acquired = chat_manager.try_acquire_lock(lock_key)
    if not lock_acquired:
        # 如果无法获取锁，说明有另一个请求正在处理
        raise HTTPException(
            status_code=429, 
            detail="当前有其他请求正在处理，请稍后再试"
        )
    
    try:
        # 更新操作时间戳
        chat_manager.update_timestamp(lock_key)
        
        # 获取指定的agent
        try:
            selected_agent = agent_manager.get_agent(agent_type)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # 首先尝试快速路径 - 跳过完整处理
        try:
            start_fast = time.time()
            fast_result = selected_agent.check_fast_cache(message, session_id)
            
            if fast_result:
                print(f"API快速路径命中: {time.time() - start_fast:.4f}s")
                
                # 在调试模式下，需要提供额外信息
                if debug:
                    # 提供模拟的执行日志
                    mock_log = [{
                        "node": "fast_cache_hit", 
                        "timestamp": time.time(), 
                        "input": message, 
                        "output": "高质量缓存命中，跳过完整处理"
                    }]
                    
                    # 尝试提取图谱数据
                    try:
                        kg_data = extract_kg_from_message(fast_result)
                    except:
                        kg_data = {"nodes": [], "links": []}
                        
                    return {
                        "answer": fast_result,
                        "execution_log": mock_log,
                        "kg_data": kg_data
                    }
                else:
                    # 标准模式直接返回答案
                    return {"answer": fast_result}
        except Exception as e:
            # 快速路径失败，继续常规流程
            print(f"快速路径检查失败: {e}")
        
        if debug:
            # 在Debug模式下使用ask_with_trace，并返回知识图谱数据
            result = selected_agent.ask_with_trace(
                message, 
                thread_id=session_id
            )
            
            # 从结果中提取知识图谱数据
            kg_data = extract_kg_from_message(result["answer"])
            
            return {
                "answer": result["answer"],
                "execution_log": result["execution_log"],
                "kg_data": kg_data
            }
        else:
            # 标准模式
            answer = selected_agent.ask(
                message, 
                thread_id=session_id
            )
            return {"answer": answer}
    except Exception as e:
        print(f"处理聊天请求时出错: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 释放锁
        chat_manager.release_lock(lock_key)
        
        # 清理过期的锁
        chat_manager.cleanup_expired_locks()


async def process_feedback(message_id: str, query: str, is_positive: bool, thread_id: str, agent_type: str = "graph_agent") -> Dict:
    """
    处理用户对回答的反馈
    
    Args:
        message_id: 消息ID
        query: 查询内容
        is_positive: 是否为正面反馈
        thread_id: 线程ID
        agent_type: Agent类型
        
    Returns:
        Dict: 反馈处理结果
    """
    try:
        # 生成锁的键
        lock_key = f"{thread_id}_{query}"
        
        # 获取锁，防止并发处理同一个查询
        with feedback_manager.get_lock(lock_key):
            # 确保agent_type存在
            try:
                selected_agent = agent_manager.get_agent(agent_type)
            except ValueError:
                agent_type = "graph_agent"  # 回退到默认agent
                print(f"未知的agent类型，使用默认值: {agent_type}")
                selected_agent = agent_manager.get_agent(agent_type)
            
            # 根据反馈进行处理
            if is_positive:
                # 标记为高质量回答
                selected_agent.mark_answer_quality(query, True, thread_id)
                action = "缓存已被标记为高质量"
            else:
                # 负面反馈 - 从缓存中移除该回答
                selected_agent.clear_cache_for_query(query, thread_id)
                action = "缓存已被清除"
                
            # 更新操作时间戳
            feedback_manager.update_timestamp(lock_key)
            
            # 清理过期的锁
            feedback_manager.cleanup_expired_locks()
            
            return {
                "status": "success",
                "action": action
            }
    except Exception as e:
        print(f"处理反馈时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))