import streamlit as st
import json
import re
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
            
            # 检查是否为deep_research_agent类型
            if st.session_state.agent_type == "deep_research_agent":
                # 检查是否有迭代数据
                if hasattr(st.session_state, 'iterations') and st.session_state.iterations:
                    # 使用后端传来的迭代数据
                    display_iterations(st.session_state.iterations)
                else:
                    # 尝试从执行日志中解析迭代信息
                    display_deep_research_trace()
            else:
                # 原有的执行轨迹显示逻辑
                for entry in st.session_state.execution_log:
                    with st.expander(f"节点: {entry['node']}", expanded=False):
                        st.markdown("**输入:**")
                        st.code(json.dumps(entry["input"], ensure_ascii=False, indent=2), language="json")
                        st.markdown("**输出:**")
                        st.code(json.dumps(entry["output"], ensure_ascii=False, indent=2), language="json")
        else:
            st.info("发送查询后将在此显示执行轨迹。")

def display_deep_research_trace():
    """显示深度研究Agent的执行轨迹"""
    if not st.session_state.execution_log:
        st.info("没有执行日志，请发送查询以生成执行轨迹")
        return
        
    # 获取deep_research节点的输出信息
    deep_research_entries = [entry for entry in st.session_state.execution_log if entry.get("node") == "deep_research"]
    
    if not deep_research_entries:
        st.info("未找到深度研究的执行轨迹，请确保选择了deep_research_agent并发送查询。")
        return
        
    # 使用最后一个deep_research条目
    entry = deep_research_entries[-1]
    
    # 尝试从output中提取迭代信息
    if "output" in entry:
        output = entry["output"]
        
        # 解析并展示迭代
        iteration_logs = parse_iteration_logs(output)
        display_iterations(iteration_logs)
    else:
        st.warning("未找到迭代信息")
        
        # 创建一个基本的迭代信息
        basic_iteration = [{
            "round": 1,
            "content": ["无法从执行日志中提取迭代信息"],
            "queries": ["原始查询"],
            "useful_info": "深度研究已完成，但无法展示详细过程"
        }]
        
        display_iterations(basic_iteration)


def parse_iteration_logs(retrieved_info):
    """
    解析迭代日志，提取各轮迭代信息
    
    Args:
        retrieved_info: 检索到的信息
        
    Returns:
        List: 迭代轮次信息
    """
    # 合并所有检索信息为一个字符串
    if isinstance(retrieved_info, list):
        # 确保元素是字符串
        retrieved_info = [str(item) for item in retrieved_info]
        full_text = "\n".join(retrieved_info)
    else:
        full_text = str(retrieved_info)
    
    # 按照迭代轮次分割文本
    iterations = []
    current_iteration = {"round": 1, "content": [], "queries": []}
    
    lines = full_text.split('\n')
    for line in lines:
        # 检测迭代轮次开始
        round_match = re.search(r'\[深度研究\]\s*开始第(\d+)轮迭代', line)
        if round_match:
            # 如果已有内容，保存前一轮
            if current_iteration["content"]:
                iterations.append(current_iteration)
            
            # 开始新一轮
            round_num = int(round_match.group(1))
            current_iteration = {"round": round_num, "content": [line], "queries": []}
        # 检测查询
        elif re.search(r'\[深度研究\]\s*执行查询:', line):
            query = re.sub(r'\[深度研究\]\s*执行查询:\s*', '', line).strip()
            current_iteration["queries"].append(query)
            current_iteration["content"].append(line)
        # 检测是否发现有用信息
        elif re.search(r'\[深度研究\]\s*发现有用信息:', line):
            current_iteration["content"].append(line)
            info = re.sub(r'\[深度研究\]\s*发现有用信息:\s*', '', line).strip()
            current_iteration["useful_info"] = info
        # 其他行
        else:
            current_iteration["content"].append(line)
    
    # 添加最后一轮
    if current_iteration["content"]:
        iterations.append(current_iteration)
    
    # 如果没有找到有效迭代，创建一个基本迭代
    if not iterations or (len(iterations) == 1 and not iterations[0].get("queries")):
        # 检查是否有全局查询和最终信息
        has_query = False
        final_info = None
        
        for i, line in enumerate(lines):
            if ">" in line and "?" in line:
                has_query = True
            if "Final Information" in line and i + 1 < len(lines):
                final_info = lines[i + 1]
        
        if has_query or final_info:
            return [{
                "round": 1,
                "content": lines,
                "queries": ["从原始查询提取"],
                "useful_info": final_info or "深度研究已完成"
            }]
    
    return iterations

def display_iterations(iterations):
    """
    显示迭代过程
    
    参数:
        iterations: 迭代数据
    """
    st.markdown("## 深度研究迭代过程")
    
    if not iterations:
        st.warning("未找到迭代信息")
        return
    
    # 使用进度条展示迭代过程
    total_iterations = len(iterations)
    progress_html = f"""
    <div style="margin: 10px 0;">
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="flex-grow: 1; height: 8px; background-color: #f0f0f0; border-radius: 4px; overflow: hidden;">
                <div style="width: 100%; height: 100%; background-color: #4CAF50; border-radius: 4px;"></div>
            </div>
            <span style="margin-left: 10px; font-weight: bold;">{total_iterations}轮迭代</span>
        </div>
    </div>
    """
    st.markdown(progress_html, unsafe_allow_html=True)
    
    # 循环显示每一轮迭代，每轮都有一个独立的expander
    for iteration in iterations:
        round_num = iteration.get("round", 0)
        
        # 创建迭代轮次的可折叠部分
        with st.expander(f"第 {round_num} 轮迭代", expanded=round_num == 1):
            # 检查内容是否为空
            if not iteration.get("content") and not iteration.get("queries"):
                st.info("此轮迭代没有详细内容")
                continue
            
            col1, col2 = st.columns([1, 1])
            
            # 左侧显示查询
            with col1:
                # 显示查询
                if iteration.get("queries"):
                    st.markdown("#### 执行的查询")
                    for query in iteration["queries"]:
                        st.code(query, language="text")
                else:
                    st.info("没有查询信息")
            
            # 右侧显示发现的信息
            with col2:
                # 显示有用信息
                if "useful_info" in iteration and iteration["useful_info"]:
                    st.markdown("#### 发现的有用信息")
                    st.success(iteration["useful_info"])
                else:
                    st.info("没有发现特别有用的信息")
            
            # 显示检索结果和其他信息
            st.markdown("#### 检索结果")
            
            # 分两列显示检索结果和其他信息
            kb_col, other_col = st.columns([1, 1])
            
            with kb_col:
                # 显示KB检索结果
                kb_results = [line for line in iteration.get("content", []) if "[KB检索]" in line]
                if kb_results:
                    st.markdown("##### 知识库检索")
                    st.code("\n".join(kb_results), language="text")
            
            with other_col:
                # 显示其他信息
                other_info = [line for line in iteration.get("content", []) 
                             if "[深度研究]" in line and "开始" not in line and "执行查询" not in line 
                             and "发现有用信息" not in line]
                if other_info:
                    st.markdown("##### 思考分析")
                    st.code("\n".join(other_info), language="text")
    
    # 添加总结信息
    if total_iterations > 0:
        st.markdown("## 最终结果")
        final_iteration = iterations[-1]
        if "useful_info" in final_iteration and final_iteration["useful_info"]:
            st.success(final_iteration["useful_info"])
        else:
            # 检查是否有"结束迭代"信息
            end_message = None
            for line in final_iteration.get("content", []):
                if "[深度研究] 没有生成新查询且已有信息，结束迭代" in line:
                    end_message = "迭代完成，已收集到足够信息"
                    break
            
            if end_message:
                st.success(end_message)
            else:
                # 从内容中提取最有价值的信息作为总结
                if final_iteration.get("content"):
                    valuable_lines = []
                    for line in final_iteration["content"]:
                        if "Final Information" in line or "最终信息" in line:
                            info_idx = final_iteration["content"].index(line)
                            if info_idx + 1 < len(final_iteration["content"]):
                                valuable_lines = final_iteration["content"][info_idx+1:info_idx+3]
                                break
                    
                    if valuable_lines:
                        st.info("\n".join(valuable_lines))
                    else:
                        st.info("深度研究已完成")

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