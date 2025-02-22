import streamlit as st
import requests
import uuid
from typing import Dict

API_URL = "http://localhost:8000"

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

def send_message(message: str) -> Dict:
    """发送聊天消息到 FastAPI 后端"""
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "message": message,
                "session_id": st.session_state.session_id,
                "debug": st.session_state.debug_mode
            }
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"服务器连接错误: {str(e)}")
        return None

def clear_chat():
    """清除聊天历史"""
    try:
        # 清除前端状态
        st.session_state.messages = []
        st.session_state.execution_log = None
        
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

def display_chat_interface():
    """显示主聊天界面"""
    st.title("💬 GraphRAG 对话")

    col1, col2 = st.columns([10, 2])
    with col2:
        if st.button("🗑️ 清除"):
            clear_chat()

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    if prompt := st.chat_input("请输入您的问题...", key="chat_input"):
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                response = send_message(prompt)
            if response:
                st.write(response["answer"])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"]
                })
                if response.get("execution_log"):
                    st.session_state.execution_log = response["execution_log"]
        
        st.rerun()

def main():
    st.set_page_config(
        page_title="GraphRAG Chat Interface",
        page_icon="🤖",
        layout="wide"
    )
    
    init_session_state()
    
    with st.sidebar:
        st.title("GraphRAG 设置")
        st.toggle("开启调试模式", value=False, key="debug_mode", help="开启后可以查看系统的调试信息")
        
        st.markdown("---")
        st.markdown("""
        ### 使用指南
        这个 GraphRAG demo 基于《悟空传》的前7章建立知识图谱来回答问题。
        
        #### 示例问题:
        1. 人物相关:
           - "《悟空传》的主要人物有哪些？"
           - "描述一下孙悟空和如来佛祖"
        
        2. 具体情节:
           - "描述一下悟空第一次见到菩提祖师的场景"
           - "唐僧和会说话的树讨论了什么？"
        
        3. 连续对话:
           - 系统会记住对话上下文
           - 你可以自然地问后续问题
        """)
    
    if st.session_state.debug_mode:
        chat_col, debug_col = st.columns([2, 1])
        with chat_col:
            display_chat_interface()
        with debug_col:
            st.title("🔍 调试信息")
            st.write(f"会话 ID: {st.session_state.session_id}")
            if st.session_state.execution_log:
                st.write("### 执行轨迹")
                for entry in st.session_state.execution_log:
                    with st.expander(f"节点: {entry['node']}", expanded=False):
                        st.write("**输入:**")
                        st.code(entry["input"])
                        st.write("**输出:**")
                        st.code(entry["output"])
    else:
        display_chat_interface()

if __name__ == "__main__":
    main()