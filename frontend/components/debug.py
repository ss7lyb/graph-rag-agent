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
    
    # 通过JS脚本直接控制标签切换
    tab_index = 0  # 默认显示执行轨迹标签
    
    if st.session_state.current_tab == "执行轨迹":
        tab_index = 0
    elif st.session_state.current_tab == "知识图谱":
        tab_index = 1
    elif st.session_state.current_tab == "源内容":
        tab_index = 2
    elif st.session_state.current_tab == "性能监控":
        tab_index = 3
    
    # 使用自定义JS自动切换到指定标签页
    tab_js = f"""
    <script>
        // 等待DOM加载完成
        document.addEventListener('DOMContentLoaded', function() {{
            setTimeout(function() {{
                // 查找所有标签按钮
                const tabs = document.querySelectorAll('[data-baseweb="tab"]');
                if (tabs.length > {tab_index}) {{
                    // 模拟点击目标标签
                    tabs[{tab_index}].click();
                }}
            }}, 100);
        }});
    </script>
    """
    
    # 只有当需要切换标签时才注入JS
    if "current_tab" in st.session_state:
        st.markdown(tab_js, unsafe_allow_html=True)