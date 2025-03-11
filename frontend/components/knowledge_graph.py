import streamlit as st
import tempfile
import os
from pyvis.network import Network
import streamlit.components.v1 as components
from utils.api import get_knowledge_graph
from frontend_config.settings import KG_COLOR_PALETTE

def visualize_knowledge_graph(kg_data):
    """使用pyvis可视化知识图谱 - 动态节点类型和颜色"""
    if not kg_data or "nodes" not in kg_data or "links" not in kg_data:
        st.warning("无法获取知识图谱数据")
        return
    
    if len(kg_data["nodes"]) == 0:
        st.info("没有找到相关的实体和关系")
        return
    
    # 添加图表设置控制
    with st.expander("图谱显示设置", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            physics_enabled = st.checkbox("启用物理引擎", 
                                       value=st.session_state.kg_display_settings["physics_enabled"],
                                       help="控制节点是否可以动态移动")
            node_size = st.slider("节点大小", 10, 50, 
                                st.session_state.kg_display_settings["node_size"],
                                help="调整节点的大小")
        
        with col2:
            edge_width = st.slider("连接线宽度", 1, 10, 
                                 st.session_state.kg_display_settings["edge_width"],
                                 help="调整连接线的宽度")
            spring_length = st.slider("弹簧长度", 50, 300, 
                                    st.session_state.kg_display_settings["spring_length"],
                                    help="调整节点之间的距离")
        
        # 更新设置
        st.session_state.kg_display_settings = {
            "physics_enabled": physics_enabled,
            "node_size": node_size,
            "edge_width": edge_width,
            "spring_length": spring_length,
            "gravity": st.session_state.kg_display_settings["gravity"]
        }
    
    # 创建网络图 - 修改背景为白色
    net = Network(height="600px", width="100%", bgcolor="#FFFFFF", font_color="#333333", directed=True)
    
    # 设置物理引擎选项，增强灵动性
    if physics_enabled:
        net.barnes_hut(
            gravity=st.session_state.kg_display_settings["gravity"], 
            central_gravity=0.5, 
            spring_length=spring_length,
            spring_strength=0.04,
            damping=0.15, 
            overlap=0.1
        )
    else:
        net.toggle_physics(False)
    
    # 使用更现代化的颜色方案
    color_palette = KG_COLOR_PALETTE
    
    # 提取所有唯一组类型
    group_types = set()
    for node in kg_data["nodes"]:
        group = node.get("group", "Unknown")
        if group:
            group_types.add(group)
    
    # 为每个组分配颜色
    group_colors = {}
    for i, group in enumerate(sorted(group_types)):
        group_colors[group] = color_palette[i % len(color_palette)]
    
    # 添加节点，使用更现代的样式
    for node in kg_data["nodes"]:
        node_id = node["id"]
        label = node.get("label", node_id)
        group = node.get("group", "Unknown")
        description = node.get("description", "")
        
        # 根据节点组类型设置颜色
        color = group_colors.get(group, "#4285F4")  # 默认使用谷歌蓝
        
        # 添加节点信息提示，改进格式
        title = f"{label}" + (f": {description}" if description else "")
        
        # 添加带有阴影和边框的节点
        net.add_node(node_id, label=label, title=title, color={"background": color, "border": "#ffffff", "highlight": {"background": color, "border": "#000000"}}, 
                    size=node_size, 
                    font={"color": "#ffffff", "size": 14, "face": "Arial"},
                    shadow={"enabled": True, "color": "rgba(0,0,0,0.2)", "size": 3})
    
    # 添加边，使用更现代的样式
    for link in kg_data["links"]:
        source = link["source"]
        target = link["target"]
        label = link.get("label", "")
        weight = link.get("weight", 1)
        
        # 根据权重设置线的粗细和不透明度
        width = edge_width * min(1 + (weight * 0.2), 3)
        
        # 使用弯曲的箭头和平滑的线条
        smooth = {"enabled": True, "type": "dynamic", "roundness": 0.5}
        
        title = label
        
        # 添加带有阴影的边
        net.add_edge(source, target, 
                    title=title, 
                    label=label, 
                    width=width, 
                    smooth=smooth,
                    color={"color": "#999999", "highlight": "#666666"},
                    shadow={"enabled": True, "color": "rgba(0,0,0,0.1)"})
    
    # 使用临时文件保存并显示网络图
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
        net.save_graph(tmp.name)
        with open(tmp.name, 'r', encoding='utf-8') as f:
            html_content = f.read()
            # 添加自定义样式，提高可读性
            html_content = html_content.replace('</head>', '''
            <style>
                .vis-network {
                    border: 1px solid #e8e8e8;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                }
                .vis-tooltip {
                    background-color: white !important;
                    color: #333 !important;
                    border: 1px solid #e0e0e0 !important;
                    border-radius: 4px !important;
                    padding: 8px !important;
                    font-family: 'Arial', sans-serif !important;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
                }
                /* 增加节点悬停动画效果 */
                .vis-node:hover {
                    transform: scale(1.1);
                    transition: all 0.3s ease;
                }
            </style>
            </head>''')
            
            # 添加额外的JavaScript，使图谱更加灵动
            html_content = html_content.replace('</body>', '''
            <script>
                // 使节点在初始加载时有一个轻微的动画效果
                setTimeout(function() {
                    network.once("stabilizationIterationsDone", function() {
                        network.setOptions({ 
                            physics: { 
                                stabilization: false,
                                barnesHut: {
                                    gravitationalConstant: -2000,  
                                    springConstant: 0.02,
                                    damping: 0.2,
                                }
                            } 
                        });
                    });
                    network.stabilize(200);
                }, 1000);
                
                // 添加鼠标悬停效果
                network.on("hoverNode", function(params) {
                    document.body.style.cursor = 'pointer';
                });
                
                network.on("blurNode", function(params) {
                    document.body.style.cursor = 'default';
                });
            </script>
            </body>''')
            
            components.html(html_content, height=600)
        
        # 清理临时文件
        try:
            os.unlink(tmp.name)
        except:
            pass
    
    # 显示图例，使用更现代的样式
    st.write("### 图例")
    
    # 创建多列显示，使用更美观的图例样式
    cols = st.columns(3)
    for i, (group, color) in enumerate(group_colors.items()):
        col_idx = i % 3
        with cols[col_idx]:
            st.markdown(
                f'<div style="display:flex;align-items:center;margin-bottom:12px">'
                f'<div style="width:20px;height:20px;border-radius:50%;background-color:{color};margin-right:10px;box-shadow:0 2px 4px rgba(0,0,0,0.1);"></div>'
                f'<span style="font-family:sans-serif;color:#333;">{group}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
    
    # 显示节点和连接数量，使用更美观的样式
    st.info(f"📊 显示 {len(kg_data['nodes'])} 个节点 和 {len(kg_data['links'])} 个关系")

def display_knowledge_graph_tab(tabs):
    """显示知识图谱标签页内容"""
    with tabs[1]:
        st.markdown('<div class="kg-controls">', unsafe_allow_html=True)

        if st.session_state.agent_type == "naive_rag_agent":
            st.info("Naive RAG 是传统的向量搜索方式，没有知识图谱的可视化。")
            return
        
        # 添加获取全局图谱/回答相关图谱的选择
        kg_display_mode = st.radio(
            "显示模式:",
            ["回答相关图谱", "全局知识图谱"],
            key="kg_display_mode",
            horizontal=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 修复：首先检查messages是否为空以及current_kg_message是否存在
        if kg_display_mode == "全局知识图谱" or "current_kg_message" not in st.session_state:
            # 获取全局图谱
            with st.spinner("加载全局知识图谱..."):
                kg_data = get_knowledge_graph(limit=100)
                if kg_data and len(kg_data.get("nodes", [])) > 0:
                    visualize_knowledge_graph(kg_data)
                else:
                    st.warning("未能加载全局知识图谱数据")
        else:
            # 显示与回答相关的图谱
            msg_idx = st.session_state.current_kg_message
            
            if (len(st.session_state.messages) > msg_idx and 
                "kg_data" in st.session_state.messages[msg_idx] and 
                len(st.session_state.messages[msg_idx]["kg_data"].get("nodes", [])) > 0):
                st.success("显示与最近回答相关的知识图谱")
                visualize_knowledge_graph(st.session_state.messages[msg_idx]["kg_data"])
            else:
                st.info("未找到与当前回答相关的知识图谱数据")
        
        # 显示节点和边的统计信息
        # 修复：添加安全检查，确保current_kg_message索引有效
        if ("current_kg_message" in st.session_state and 
            len(st.session_state.messages) > st.session_state.current_kg_message and
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