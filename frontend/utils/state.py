import streamlit as st
import uuid
from frontend_config.settings import DEFAULT_KG_SETTINGS

def init_session_state():
    """初始化会话状态变量"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    if 'execution_log' not in st.session_state:
        st.session_state.execution_log = None
    if 'agent_type' not in st.session_state:
        st.session_state.agent_type = "graph_agent"  # 默认使用graph_agent
    if 'kg_data' not in st.session_state:
        st.session_state.kg_data = None
    if 'source_content' not in st.session_state:
        st.session_state.source_content = None
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "执行轨迹"
    if 'kg_display_settings' not in st.session_state:
        st.session_state.kg_display_settings = DEFAULT_KG_SETTINGS
    if 'feedback_given' not in st.session_state:
        st.session_state.feedback_given = set()
    if 'feedback_in_progress' not in st.session_state:
        st.session_state.feedback_in_progress = False
    if 'processing_lock' not in st.session_state:
        st.session_state.processing_lock = False
    if 'current_kg_message' not in st.session_state:
        st.session_state.current_kg_message = None