import streamlit as st
from utils.api import get_knowledge_graph
from .visualization import visualize_knowledge_graph

def display_knowledge_graph_tab(tabs):
    """显示知识图谱标签页内容 - 懒加载"""
    with tabs[1]:
        st.markdown('<div class="kg-controls">', unsafe_allow_html=True)

        # 检查当前agent类型，deep_research_agent和naive_rag_agent都禁用图谱
        if st.session_state.agent_type == "naive_rag_agent":
            st.info("Naive RAG 是传统的向量搜索方式，没有知识图谱的可视化。")
            return
        elif st.session_state.agent_type == "deep_research_agent":
            st.info("Deep Research Agent 专注于深度推理过程，没有知识图谱的可视化。请查看执行轨迹标签页了解详细推理过程。")
            return
        
        # 添加获取全局图谱/回答相关图谱的选择
        kg_display_mode = st.radio(
            "显示模式:",
            ["回答相关图谱", "全局知识图谱"],
            key="kg_display_mode",
            horizontal=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 使用会话状态跟踪是否需要加载图谱
        # 如果是第一次访问标签页或者切换了显示模式，才需要加载
        should_load_kg = False
        
        # 检查是否是第一次切换到这个标签或显示模式改变
        if "current_tab" in st.session_state and st.session_state.current_tab == "知识图谱":
            if "last_kg_mode" not in st.session_state or st.session_state.last_kg_mode != kg_display_mode:
                should_load_kg = True
                st.session_state.last_kg_mode = kg_display_mode
        
        # 如果不需要加载，检查是否已有缓存的图谱数据
        if not should_load_kg:
            if kg_display_mode == "回答相关图谱" and "kg_related_cached" in st.session_state:
                st.success(st.session_state.kg_related_message)
                try:
                    st.session_state.kg_related_cached()
                except Exception as e:
                    st.error(f"显示图谱时出错: {str(e)}")
                    # 重置相关缓存
                    if "kg_related_cached" in st.session_state:
                        del st.session_state.kg_related_cached
                    # 尝试显示全局图谱
                    st.warning("无法显示相关图谱，尝试加载全局知识图谱...")
                    with st.spinner("加载全局知识图谱..."):
                        kg_data = get_knowledge_graph(limit=100)
                        if kg_data and len(kg_data.get("nodes", [])) > 0:
                            visualize_knowledge_graph(kg_data)
                return
            elif kg_display_mode == "全局知识图谱" and "kg_global_cached" in st.session_state:
                st.success("显示全局知识图谱")
                try:
                    st.session_state.kg_global_cached()
                except Exception as e:
                    st.error(f"显示全局图谱时出错: {str(e)}")
                    # 重置相关缓存
                    if "kg_global_cached" in st.session_state:
                        del st.session_state.kg_global_cached
                    # 重新加载全局图谱
                    with st.spinner("重新加载全局知识图谱..."):
                        kg_data = get_knowledge_graph(limit=100)
                        if kg_data and len(kg_data.get("nodes", [])) > 0:
                            visualize_knowledge_graph(kg_data)
                return
        
        # 需要重新加载图谱数据
        if kg_display_mode == "回答相关图谱" and "current_kg_message" in st.session_state and st.session_state.current_kg_message is not None:
            msg_idx = st.session_state.current_kg_message
            
            # 安全地检查索引是否有效以及kg_data是否存在
            if (0 <= msg_idx < len(st.session_state.messages) and 
                "kg_data" in st.session_state.messages[msg_idx] and 
                st.session_state.messages[msg_idx]["kg_data"] is not None and
                len(st.session_state.messages[msg_idx]["kg_data"].get("nodes", [])) > 0):
                
                # 获取相关回答的消息内容前20个字符用于显示
                msg_preview = st.session_state.messages[msg_idx]["content"][:20] + "..."
                st.success(f"显示与回答「{msg_preview}」相关的知识图谱")
                
                # 缓存消息以供后续使用
                st.session_state.kg_related_message = f"显示与回答「{msg_preview}」相关的知识图谱"
                
                # 使用缓存机制记住可视化函数
                def display_cached_kg():
                    if msg_idx >= 0 and msg_idx < len(st.session_state.messages) and "kg_data" in st.session_state.messages[msg_idx]:
                        visualize_knowledge_graph(st.session_state.messages[msg_idx]["kg_data"])
                    else:
                        st.error("无法显示知识图谱：消息索引无效或缺少图谱数据")
                        # 重置相关状态
                        if "kg_related_cached" in st.session_state:
                            del st.session_state.kg_related_cached
                        if "current_kg_message" in st.session_state:
                            st.session_state.current_kg_message = None
                st.session_state.kg_related_cached = display_cached_kg
                
                # 显示图谱
                try:
                    display_cached_kg()
                except Exception as e:
                    st.error(f"显示图谱时出错: {str(e)}")
                    # 尝试显示全局图谱
                    st.warning("尝试加载全局知识图谱...")
                    with st.spinner("加载全局知识图谱..."):
                        kg_data = get_knowledge_graph(limit=100)
                        if kg_data and len(kg_data.get("nodes", [])) > 0:
                            visualize_knowledge_graph(kg_data)
            else:
                st.info("未找到与当前回答相关的知识图谱数据")
                # 如果没有相关的图谱数据，默认显示全局图谱
                with st.spinner("加载全局知识图谱..."):
                    kg_data = get_knowledge_graph(limit=100)
                    if kg_data and len(kg_data.get("nodes", [])) > 0:
                        st.warning("显示全局知识图谱（未找到回答相关图谱）")
                        
                        # 缓存全局图谱以供后续使用
                        def display_cached_global_kg():
                            visualize_knowledge_graph(kg_data)
                        st.session_state.kg_global_cached = display_cached_global_kg
                        
                        # 显示图谱
                        display_cached_global_kg()
                    else:
                        st.warning("未能加载任何知识图谱数据")
        else:
            # 获取全局图谱
            with st.spinner("加载全局知识图谱..."):
                kg_data = get_knowledge_graph(limit=100)
                if kg_data and len(kg_data.get("nodes", [])) > 0:
                    # 缓存全局图谱以供后续使用
                    def display_cached_global_kg():
                        visualize_knowledge_graph(kg_data)
                    st.session_state.kg_global_cached = display_cached_global_kg
                    
                    # 显示图谱
                    display_cached_global_kg()
                else:
                    st.warning("未能加载全局知识图谱数据")
        
        # 显示当前图谱的节点和关系数量
        try:
            if ("current_kg_message" in st.session_state and 
                st.session_state.current_kg_message is not None and
                0 <= st.session_state.current_kg_message < len(st.session_state.messages) and
                "kg_data" in st.session_state.messages[st.session_state.current_kg_message]):
                
                kg_data = st.session_state.messages[st.session_state.current_kg_message]["kg_data"]
                if kg_data and len(kg_data.get("nodes", [])) > 0:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("节点数量", len(kg_data["nodes"]))
                    with col2:
                        st.metric("关系数量", len(kg_data["links"]))
            elif kg_display_mode == "回答相关图谱":
                st.info("在调试模式下发送查询获取相关的知识图谱")
        except Exception as e:
            st.error(f"显示图谱统计信息时出错: {str(e)}")