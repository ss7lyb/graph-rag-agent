import streamlit as st
import requests
import uuid
import json
import pandas as pd
import networkx as nx
from typing import Dict, List, Any
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
import os
import re

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
    if 'agent_type' not in st.session_state:
        st.session_state.agent_type = "graph_agent"  # 默认使用graph_agent
    if 'kg_data' not in st.session_state:
        st.session_state.kg_data = None
    if 'source_content' not in st.session_state:
        st.session_state.source_content = None
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "执行轨迹"
    if 'kg_display_settings' not in st.session_state:
        st.session_state.kg_display_settings = {
            "physics_enabled": True,
            "node_size": 25,
            "edge_width": 2,
            "spring_length": 150,
            "gravity": -5000
        }

def send_message(message: str) -> Dict:
    """发送聊天消息到 FastAPI 后端"""
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "message": message,
                "session_id": st.session_state.session_id,
                "debug": st.session_state.debug_mode,
                "agent_type": st.session_state.agent_type
            },
            timeout=60  # 增加超时时间
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"服务器连接错误: {str(e)}")
        return None

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

def get_knowledge_graph_from_message(message: str) -> Dict:
    """从AI响应中提取知识图谱数据"""
    try:
        response = requests.get(
            f"{API_URL}/knowledge_graph_from_message",
            params={"message": message},
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

def clear_chat():
    """清除聊天历史"""
    try:
        # 清除前端状态
        st.session_state.messages = []
        st.session_state.execution_log = None
        st.session_state.kg_data = None
        st.session_state.source_content = None
        
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

def extract_source_ids(answer: str) -> List[str]:
    """从回答中提取引用的源ID"""
    source_ids = []
    
    # 提取Chunks IDs
    chunks_pattern = r"Chunks':\s*\[([^\]]*)\]"
    matches = re.findall(chunks_pattern, answer)
    
    if matches:
        for match in matches:
            # 处理带引号的ID
            quoted_ids = re.findall(r"'([^']*)'", match)
            if quoted_ids:
                source_ids.extend(quoted_ids)
            else:
                # 处理不带引号的ID
                ids = [id.strip() for id in match.split(',') if id.strip()]
                source_ids.extend(ids)
    
    # 去重
    return list(set(source_ids))

def visualize_knowledge_graph(kg_data: Dict) -> None:
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
    
    # 创建网络图
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white", directed=True)
    
    # 设置物理引擎选项
    if physics_enabled:
        net.barnes_hut(
            gravity=st.session_state.kg_display_settings["gravity"], 
            central_gravity=0.3, 
            spring_length=spring_length
        )
    else:
        net.toggle_physics(False)
    
    # 动态生成颜色映射
    color_palette = [
        "#1f77b4",  # 蓝色
        "#ff7f0e",  # 橙色
        "#2ca02c",  # 绿色
        "#d62728",  # 红色
        "#9467bd",  # 紫色
        "#8c564b",  # 棕色
        "#e377c2",  # 粉色
        "#7f7f7f",  # 灰色
        "#bcbd22",  # 黄绿色
        "#17becf"   # 青色
    ]
    
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
    
    # 添加节点
    for node in kg_data["nodes"]:
        node_id = node["id"]
        label = node.get("label", node_id)
        group = node.get("group", "Unknown")
        description = node.get("description", "")
        
        # 根据节点组类型设置颜色
        color = group_colors.get(group, "#1f77b4")  # 默认蓝色
        
        # 添加节点信息提示
        title = f"{label}: {description}" if description else label
        
        net.add_node(node_id, label=label, title=title, color=color, size=node_size)
    
    # 添加边
    for link in kg_data["links"]:
        source = link["source"]
        target = link["target"]
        label = link.get("label", "")
        weight = link.get("weight", 1)
        
        # 根据权重设置线的粗细
        width = edge_width * min(1 + (weight * 0.2), 3)
        
        # 使用弯曲的箭头
        smooth = True
        
        net.add_edge(source, target, title=label, label=label, width=width, smooth=smooth)
    
    # 使用临时文件保存并显示网络图
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
        net.save_graph(tmp.name)
        with open(tmp.name, 'r', encoding='utf-8') as f:
            html_content = f.read()
            # 添加自定义样式，提高可读性
            html_content = html_content.replace('</head>', '''
            <style>
                .vis-network {
                    border: 1px solid #444;
                    border-radius: 8px;
                }
                .vis-tooltip {
                    background-color: #333 !important;
                    color: #fff !important;
                    border: 1px solid #555 !important;
                    border-radius: 4px !important;
                    padding: 8px !important;
                    font-family: 'Arial', sans-serif !important;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.3) !important;
                }
            </style>
            </head>''')
            components.html(html_content, height=600)
        
        # 清理临时文件
        try:
            os.unlink(tmp.name)
        except:
            pass
    
    # 显示图例
    st.write("### 图例")
    
    # 创建多列显示
    cols = st.columns(3)
    for i, (group, color) in enumerate(group_colors.items()):
        col_idx = i % 3
        with cols[col_idx]:
            st.markdown(
                f'<div style="display:flex;align-items:center;margin-bottom:8px">'
                f'<div style="width:20px;height:20px;border-radius:50%;background-color:{color};margin-right:8px"></div>'
                f'<span>{group}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
    
    # 显示节点和连接数量
    st.info(f"显示 {len(kg_data['nodes'])} 个节点 和 {len(kg_data['links'])} 个关系")

def insert_example_question(question: str):
    """将示例问题插入聊天输入框"""
    st.session_state.chat_input = question

def display_source_content(content: str):
    """更好地显示源内容"""
    st.markdown("""
    <style>
    .source-content {
        white-space: pre-wrap;
        overflow-x: auto;
        font-family: monospace;
        line-height: 1.6;
        background-color: #f5f5f5;
        border-radius: 5px;
        padding: 15px;
        max-height: 600px;
        overflow-y: auto;
        border: 1px solid #e1e4e8;
        color: #24292e;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 将换行符转换为HTML换行，确保格式正确
    formatted_content = content.replace("\n", "<br>")
    st.markdown(f'<div class="source-content">{formatted_content}</div>', unsafe_allow_html=True)

def custom_css():
    """添加自定义CSS样式"""
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4b9bff;
        color: white;
    }
    .agent-selector {
        padding: 10px;
        margin-bottom: 20px;
        border-radius: 5px;
        background-color: #f7f7f7;
    }
    .chat-container {
        border-radius: 10px;
        background-color: white;
        padding: 10px;
        height: calc(100vh - 250px);
        overflow-y: auto;
        margin-bottom: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    .debug-container {
        border-radius: 10px;
        background-color: white;
        height: calc(100vh - 120px);
        overflow-y: auto;
        padding: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    .example-question {
        background-color: #f7f7f7;
        padding: 8px;
        border-radius: 4px;
        margin: 5px 0;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .example-question:hover {
        background-color: #e6e6e6;
    }
    .settings-bar {
        padding: 10px;
        background-color: #f7f7f7;
        border-radius: 5px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    /* 源内容样式 - 改进版 */
    .source-content-container {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin-top: 10px;
        border: 1px solid #e0e0e0;
    }
    .source-content {
        white-space: pre-wrap;
        word-wrap: break-word;
        background-color: #f5f5f5;
        padding: 16px;
        border-radius: 4px;
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        font-size: 14px;
        line-height: 1.6;
        overflow-x: auto;
        color: #24292e;
        max-height: 600px;
        overflow-y: auto;
        border: 1px solid #e1e4e8;
    }
    /* 调试信息样式 */
    .debug-header {
        background-color: #eef2f5;
        padding: 10px 15px;
        border-radius: 5px;
        margin-bottom: 15px;
        border-left: 4px solid #4b9bff;
    }
    /* 知识图谱控制面板 */
    .kg-controls {
        background-color: #f8f9fa;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 15px;
        border: 1px solid #e6e6e6;
    }
    /* 按钮悬停效果 */
    button:hover {
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        transition: all 0.3s cubic-bezier(.25,.8,.25,1);
    }
    /* 源内容按钮样式 */
    .view-source-button {
        background-color: #f1f8ff;
        border: 1px solid #c8e1ff;
        color: #0366d6;
        border-radius: 6px;
        padding: 4px 8px;
        font-size: 12px;
        margin: 4px;
    }
    .view-source-button:hover {
        background-color: #dbedff;
    }
    </style>
    """, unsafe_allow_html=True)

def display_chat_interface():
    """显示主聊天界面"""
    st.title("GraphRAG 对话系统")
    
    # 设置栏
    with st.container():
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            # 使用不同的key: header_agent_type
            agent_type = st.selectbox(
                "选择 Agent 类型",
                options=["graph_agent", "hybrid_agent"],
                key="header_agent_type",
                help="选择不同的Agent以体验不同的检索策略",
                index=0 if st.session_state.agent_type == "graph_agent" else 1
            )
            # 更新全局agent_type
            st.session_state.agent_type = agent_type
        
        with col2:
            debug_mode = st.toggle("调试模式", value=st.session_state.debug_mode, key="header_debug_mode")
            # 更新全局debug_mode
            st.session_state.debug_mode = debug_mode
        
        with col3:
            st.button("🗑️ 清除聊天", on_click=clear_chat)
    
    # 分隔线
    st.markdown("---")
    
    # 聊天区域
    chat_container = st.container()
    with chat_container:
        # 显示现有消息
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                
                # 如果是AI回答且有源内容引用，显示查看源内容按钮
                if msg["role"] == "assistant" and st.session_state.debug_mode:
                    source_ids = extract_source_ids(msg["content"])
                    if source_ids:
                        with st.expander("查看引用源文本", expanded=False):
                            for source_id in source_ids:
                                if st.button(f"加载源文本 {source_id}", key=f"src_{source_id}_{i}"):
                                    with st.spinner(f"加载源文本 {source_id}..."):
                                        source_data = get_source_content(source_id)
                                        if source_data and "content" in source_data:
                                            st.session_state.source_content = source_data["content"]
                                            st.session_state.current_tab = "源内容"  # 自动切换到源内容标签
                                            st.rerun()
                    
                    # 如果是最后一条AI消息，添加自动提取图谱按钮
                    if i == len(st.session_state.messages) - 1:
                        if st.button("提取知识图谱", key=f"extract_kg_{i}"):
                            with st.spinner("提取知识图谱数据..."):
                                kg_data = get_knowledge_graph_from_message(msg["content"])
                                if kg_data and len(kg_data.get("nodes", [])) > 0:
                                    st.session_state.kg_data = kg_data
                                    st.session_state.current_tab = "知识图谱"  # 自动切换到知识图谱标签
                                    st.rerun()
        
        # 处理新消息
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
                        
                    # 从回答中提取知识图谱数据
                    if st.session_state.debug_mode:
                        try:
                            with st.spinner("提取知识图谱数据..."):
                                # 优先使用后端返回的kg_data
                                kg_data = response.get("kg_data")
                                
                                # 如果后端没有返回kg_data，尝试从回答中提取
                                if not kg_data or len(kg_data.get("nodes", [])) == 0:
                                    kg_data = get_knowledge_graph_from_message(response["answer"])
                                
                                if kg_data and len(kg_data.get("nodes", [])) > 0:
                                    st.session_state.kg_data = kg_data
                                    st.session_state.current_tab = "知识图谱"  # 自动切换到知识图谱标签
                        except Exception as e:
                            print(f"提取知识图谱失败: {e}")
            
            st.rerun()

def display_knowledge_graph_tab(tabs):
    """显示知识图谱标签页内容"""
    with tabs[1]:
        st.markdown('<div class="kg-controls">', unsafe_allow_html=True)
        
        # 添加获取全局图谱/回答相关图谱的选择
        kg_display_mode = st.radio(
            "显示模式:",
            ["回答相关图谱", "全局知识图谱"],
            key="kg_display_mode",
            horizontal=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if kg_display_mode == "全局知识图谱" or not st.session_state.kg_data:
            # 获取全局图谱
            with st.spinner("加载全局知识图谱..."):
                kg_data = get_knowledge_graph(limit=100)
                if kg_data and len(kg_data.get("nodes", [])) > 0:
                    visualize_knowledge_graph(kg_data)
                else:
                    st.warning("未能加载全局知识图谱数据")
        else:
            # 显示与回答相关的图谱
            if st.session_state.kg_data and len(st.session_state.kg_data.get("nodes", [])) > 0:
                st.success("显示与最近回答相关的知识图谱")
                visualize_knowledge_graph(st.session_state.kg_data)
            else:
                st.info("未找到与当前回答相关的知识图谱数据")
        
        # 显示节点和边的统计信息
        if st.session_state.kg_data and len(st.session_state.kg_data.get("nodes", [])) > 0:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("节点数量", len(st.session_state.kg_data["nodes"]))
            with col2:
                st.metric("关系数量", len(st.session_state.kg_data["links"]))
        elif kg_display_mode == "回答相关图谱":
            st.info("在调试模式下发送查询获取相关的知识图谱")

def display_source_content_tab(tabs):
    """显示源内容标签页内容"""
    with tabs[2]:
        if st.session_state.source_content:
            st.markdown('<div class="source-content-container">', unsafe_allow_html=True)
            display_source_content(st.session_state.source_content)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("点击AI回答中的'查看源内容'按钮查看源文本")

def main():
    # 页面配置
    st.set_page_config(
        page_title="GraphRAG Chat Interface",
        page_icon="🤖",
        layout="wide"
    )
    
    # 初始化会话状态
    init_session_state()
    
    # 添加自定义CSS
    custom_css()
    
    # 页面布局: 侧边栏和主区域
    with st.sidebar:
        st.title("📚 GraphRAG")
        st.markdown("---")
        
        # Agent选择部分
        st.header("Agent 选择")
        agent_type = st.radio(
            "选择检索策略:",
            ["graph_agent", "hybrid_agent"],
            index=0 if st.session_state.agent_type == "graph_agent" else 1,
            help="graph_agent：使用知识图谱的局部与全局搜索；hybrid_agent：使用混合搜索方式",
            key="sidebar_agent_type"
        )
        # 更新全局agent_type
        st.session_state.agent_type = agent_type
        
        st.markdown("---")
        
        # 调试选项
        st.header("调试选项")
        debug_mode = st.checkbox("启用调试模式", 
                               value=st.session_state.debug_mode, 
                               key="sidebar_debug_mode",
                               help="显示执行轨迹、知识图谱和源内容")
        # 更新全局debug_mode
        st.session_state.debug_mode = debug_mode
        
        st.markdown("---")
        
        # 示例问题部分
        st.header("示例问题")
        example_questions = [
            "《悟空传》的主要人物有哪些？",
            "唐僧和会说话的树讨论了什么？",
            "孙悟空跟女妖之间有什么故事？",
            "他最后的选择是什么？"
        ]
        
        for i, question in enumerate(example_questions):
            st.markdown(
                f"""<div class="example-question" 
                    onclick="document.querySelector('.stChatInputContainer input').value='{question}';
                    document.querySelector('.stChatInputContainer button').click();">
                    {question}
                </div>""", 
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        
        # 项目信息
        st.markdown("""
        ### 关于
        这个 GraphRAG 演示基于《悟空传》的前7章建立知识图谱，使用不同的Agent策略回答问题。
        
        **调试模式**可查看:
        - 执行轨迹
        - 知识图谱可视化
        - 原始文本内容
        """)
        
        # 重置按钮
        if st.button("🗑️ 清除对话历史", key="clear_chat"):
            clear_chat()
    
    # 主区域布局
    if st.session_state.debug_mode:
        # 调试模式下的布局（左侧聊天，右侧调试信息）
        col1, col2 = st.columns([5, 4])
        
        with col1:
            display_chat_interface()
            
        with col2:
            st.subheader("🔍 调试信息")
            
            # 创建标签页用于不同类型的调试信息
            tabs = st.tabs(["执行轨迹", "知识图谱", "源内容"])
            
            # 执行轨迹标签
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
            
            # 知识图谱标签
            display_knowledge_graph_tab(tabs)
            
            # 源内容标签
            display_source_content_tab(tabs)
            
            # 自动选择标签页
            if st.session_state.current_tab == "执行轨迹":
                tabs[0].active = True
            elif st.session_state.current_tab == "知识图谱":
                tabs[1].active = True
            elif st.session_state.current_tab == "源内容":
                tabs[2].active = True
    else:
        # 非调试模式下的布局（仅聊天界面）
        display_chat_interface()

if __name__ == "__main__":
    main()