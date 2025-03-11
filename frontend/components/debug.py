import streamlit as st
import json
from utils.helpers import display_source_content
from utils.performance import display_performance_stats, clear_performance_data
from components.knowledge_graph import display_knowledge_graph_tab

def display_source_content_tab(tabs):
    """显示源内容标签页内容"""
    with tabs[2]:
        if st.session_state.source_content:
            st.markdown('<div class="source-content-container">', unsafe_allow_html=True)
            display_source_content(st.session_state.source_content)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("点击AI回答中的'查看源内容'按钮查看源文本")

def display_execution_trace_tab(tabs):
    """显示执行轨迹标签页内容"""
    with tabs[0]:
        if st.session_state.execution_log:
            st.markdown(f'<div class="debug-header">会话 ID: {st.session_state.session_id}</div>', unsafe_allow_html=True)
            for entry in st.session_state.execution_log:
                with st.expander(f"节点: {entry['node']}", expanded=False):
                    st.markdown("**输入:**")
                    st.code(json.dumps(entry["input"], ensure_ascii=False, indent=2), language="json")
                    st.markdown("**输出:**")
                    st.code(json.dumps(entry["output"], ensure_ascii=False, indent=2), language="json")
        else:
            st.info("发送查询后将在此显示执行轨迹。")

def add_performance_tab(tabs):
    """添加性能监控标签页"""
    with tabs[3]:  # 第四个标签页
        st.markdown('<div class="debug-header">性能统计</div>', unsafe_allow_html=True)
        display_performance_stats()
        
        # 添加清除性能数据的按钮
        if st.button("清除性能数据"):
            clear_performance_data()
            st.rerun()

def display_debug_panel():
    """显示调试面板"""
    st.subheader("🔍 调试信息")
    
    # 创建标签页用于不同类型的调试信息
    tabs = st.tabs(["执行轨迹", "知识图谱", "源内容", "性能监控"])
    
    # 执行轨迹标签
    display_execution_trace_tab(tabs)
    
    # 知识图谱标签
    display_knowledge_graph_tab(tabs)
    
    # 源内容标签
    display_source_content_tab(tabs)
    
    # 性能监控标签
    add_performance_tab(tabs)
    
    # 自动选择标签页
    if st.session_state.current_tab == "执行轨迹":
        tabs[0].activate = True
    elif st.session_state.current_tab == "知识图谱":
        tabs[1].activate = True
    elif st.session_state.current_tab == "源内容":
        tabs[2].activate = True
    elif st.session_state.current_tab == "性能监控":
        tabs[3].activate = True