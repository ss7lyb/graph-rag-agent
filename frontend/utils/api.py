import time
import uuid
import requests
import streamlit as st
from typing import Dict
from frontend_config.settings import API_URL

def send_message(message: str) -> Dict:
    """发送聊天消息到 FastAPI 后端，带性能监控"""
    start_time = time.time()
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "message": message,
                "session_id": st.session_state.session_id,
                "debug": st.session_state.debug_mode,
                "agent_type": st.session_state.agent_type
            },
            timeout=120  # 增加超时时间
        )
        
        # 记录性能
        duration = time.time() - start_time
        print(f"前端API调用耗时: {duration:.4f}s")
        
        # 在会话中保存性能数据
        if 'performance_metrics' not in st.session_state:
            st.session_state.performance_metrics = []
            
        st.session_state.performance_metrics.append({
            "operation": "send_message",
            "duration": duration,
            "timestamp": time.time(),
            "message_length": len(message)
        })
        
        return response.json()
    except requests.exceptions.RequestException as e:
        # 记录错误性能
        duration = time.time() - start_time
        print(f"前端API调用错误: {str(e)} ({duration:.4f}s)")
        
        st.error(f"服务器连接错误: {str(e)}")
        return None

def send_feedback(message_id: str, query: str, is_positive: bool, thread_id: str, agent_type: str = "graph_agent"):
    """向后端发送用户反馈 - 增加防抖和错误处理，带性能监控"""
    start_time = time.time()
    try:
        # 确保 agent_type 有值
        if not agent_type:
            agent_type = "graph_agent"
            
        response = requests.post(
            f"{API_URL}/feedback",
            json={
                "message_id": message_id,
                "query": query,
                "is_positive": is_positive,
                "thread_id": thread_id,
                "agent_type": agent_type  # 确保这个字段被包含在请求中
            },
            timeout=10
        )
        
        # 记录性能
        duration = time.time() - start_time
        print(f"前端反馈API调用耗时: {duration:.4f}s")
        
        # 在会话中保存性能数据
        if 'performance_metrics' not in st.session_state:
            st.session_state.performance_metrics = []
            
        st.session_state.performance_metrics.append({
            "operation": "send_feedback",
            "duration": duration,
            "timestamp": time.time(),
            "is_positive": is_positive
        })
        
        # 记录和返回响应
        try:
            return response.json()
        except:
            return {"status": "error", "action": "解析响应失败"}
    except requests.exceptions.RequestException as e:
        # 记录错误性能
        duration = time.time() - start_time
        print(f"前端反馈API调用错误: {str(e)} ({duration:.4f}s)")
        
        st.error(f"发送反馈时出错: {str(e)}")
        return {"status": "error", "action": str(e)}

def get_knowledge_graph(limit: int = 100, query: str = None) -> Dict:
    """获取知识图谱数据"""
    try:
        params = {"limit": limit}
        if query:
            params["query"] = query
            
        response = requests.get(
            f"{API_URL}/knowledge_graph",
            params=params,
            timeout=30
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"获取知识图谱时出错: {str(e)}")
        return {"nodes": [], "links": []}

def get_knowledge_graph_from_message(message: str, query: str = None):
    """从AI响应中提取知识图谱数据"""
    try:
        params = {"message": message}
        if query:
            params["query"] = query
            
        response = requests.get(
            f"{API_URL}/knowledge_graph_from_message",
            params=params,
            timeout=30
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"从响应提取知识图谱时出错: {str(e)}")
        return {"nodes": [], "links": []}

def get_source_content(source_id: str) -> Dict:
    """获取源内容"""
    try:
        response = requests.post(
            f"{API_URL}/source",
            json={"source_id": source_id},
            timeout=30
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"获取源内容时出错: {str(e)}")
        return None

def get_source_file_info(source_id: str) -> dict:
    """获取源ID对应的文件信息
    
    Args:
        source_id: 源ID
        
    Returns:
        Dict: 包含文件名等信息的字典
    """
    try:
        response = requests.post(
            f"{API_URL}/source_info",
            json={"source_id": source_id},
            timeout=10
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"获取源文件信息时出错: {str(e)}")
        return {"file_name": f"源文本 {source_id}"}

def clear_chat():
    """清除聊天历史"""
    try:
        # 清除前端状态
        st.session_state.messages = []
        st.session_state.execution_log = None
        st.session_state.kg_data = None
        st.session_state.source_content = None
        
        # 重要：也要清除current_kg_message
        if 'current_kg_message' in st.session_state:
            del st.session_state.current_kg_message
        
        # 清除后端状态
        response = requests.post(
            f"{API_URL}/clear",
            json={"session_id": st.session_state.session_id}
        )
        
        if response.status_code != 200:
            st.error("清除后端对话历史失败")
            return
            
        # 重新生成会话ID
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
        
    except Exception as e:
        st.error(f"清除对话时发生错误: {str(e)}")