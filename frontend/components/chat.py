import streamlit as st
import uuid
import re
from utils.api import send_message, send_feedback, get_source_content, get_knowledge_graph_from_message, get_source_file_info_batch, clear_chat
from utils.helpers import extract_source_ids

def reset_processing_lock():
    """重置处理锁状态"""
    st.session_state.processing_lock = False

def display_chat_interface():
    """显示主聊天界面"""
    st.title("GraphRAG 对话系统")
    
    # 确保锁变量存在并设初始值，如果不存在的话
    if "processing_lock" not in st.session_state:
        st.session_state.processing_lock = False
    
    # 设置栏
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # 添加重置锁的功能到 selectbox 的 on_change 参数
            previous_agent = st.session_state.agent_type
            agent_type = st.selectbox(
                "选择 Agent 类型",
                options=["graph_agent", "hybrid_agent", "naive_rag_agent", "deep_research_agent"],
                key="header_agent_type",
                help="选择不同的Agent以体验不同的检索策略",
                index=0 if st.session_state.agent_type == "graph_agent" 
                        else (1 if st.session_state.agent_type == "hybrid_agent" 
                             else (2 if st.session_state.agent_type == "naive_rag_agent"
                                  else 3)),
                on_change=reset_processing_lock
            )
            
            # 检查是否切换了agent类型
            if previous_agent != agent_type:
                # 切换agent类型时重置锁
                st.session_state.processing_lock = False
                
            st.session_state.agent_type = agent_type
            
            # 添加思考过程切换 - 仅当选择 deep_research_agent 时显示
            if agent_type == "deep_research_agent":
                show_thinking = st.checkbox("显示推理过程", 
                          value=st.session_state.get("show_thinking", False),
                          key="header_show_thinking",
                          help="显示AI的思考过程",
                          on_change=reset_processing_lock)
                st.session_state.show_thinking = show_thinking

                use_deeper = st.checkbox("使用增强版研究工具", 
                                        value=st.session_state.get("use_deeper_tool", True),
                                        key="header_use_deeper",
                                        help="启用社区感知和知识图谱增强",
                                        on_change=reset_processing_lock)
                st.session_state.use_deeper_tool = use_deeper
    
        with col2:
            # 修改清除聊天按钮，添加重置锁的功能
            st.button("🗑️ 清除聊天", key="header_clear_chat", on_click=clear_chat_with_lock_reset)
    
    # 分隔线
    st.markdown("---")
    
    # 如果当前有正在处理的请求，显示警告
    if st.session_state.processing_lock:
        st.warning("请等待当前操作完成...")
        # 添加强制重置锁的按钮
        if st.button("强制重置处理状态", key="force_reset_lock"):
            st.session_state.processing_lock = False
            st.rerun()
    
    # 聊天区域
    chat_container = st.container()
    with chat_container:
        # 显示现有消息
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                # 获取要显示的内容
                content = msg["content"]
                
                # 处理deep_research_agent的思考过程
                if msg["role"] == "assistant":
                    # 判断是否需要显示思考过程
                    show_thinking = (st.session_state.agent_type == "deep_research_agent" and 
                                    st.session_state.get("show_thinking", False))
                    
                    # 优先使用raw_thinking字段
                    if "raw_thinking" in msg and show_thinking:
                        # 提取思考过程
                        thinking_process = msg["raw_thinking"]
                        answer_content = msg.get("processed_content", content)
                        
                        # 格式化思考过程，使用引用格式
                        thinking_lines = thinking_process.split('\n')
                        quoted_thinking = '\n'.join([f"> {line}" for line in thinking_lines])
                        
                        # 显示思考过程
                        st.markdown(quoted_thinking)
                        
                        # 添加两行空行间隔
                        st.markdown("\n\n")
                        
                        # 显示答案
                        st.markdown(answer_content)
                    # 检查是否有<think>标签
                    elif "<think>" in content and "</think>" in content:
                        # 提取<think>标签中的内容
                        thinking_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                        
                        if thinking_match:
                            thinking_process = thinking_match.group(1)
                            # 移除思考过程，保留答案
                            answer_content = content.replace(f"<think>{thinking_process}</think>", "").strip()
                            
                            if show_thinking:
                                # 显示思考过程（仅当show_thinking为True时）
                                # 格式化思考过程，使用引用格式
                                thinking_lines = thinking_process.split('\n')
                                quoted_thinking = '\n'.join([f"> {line}" for line in thinking_lines])
                                
                                # 显示思考过程
                                st.markdown(quoted_thinking)
                                
                                # 添加两行空行间隔
                                st.markdown("\n\n")
                                
                                # 显示答案
                                st.markdown(answer_content)
                            else:
                                # 只显示答案部分（不显示思考过程）
                                st.markdown(answer_content)
                        else:
                            # 如果提取失败，显示完整内容但移除可能的<think>标签
                            cleaned_content = re.sub(r'<think>|</think>', '', content)
                            st.markdown(cleaned_content)
                    else:
                        # 普通回答，无思考过程
                        st.markdown(content)
                else:
                    # 普通消息直接显示
                    st.markdown(content)
                
                # 为AI回答添加反馈按钮和源内容引用
                if msg["role"] == "assistant":
                    # 生成一个唯一的消息ID (如果之前没有)
                    if "message_id" not in msg:
                        msg["message_id"] = str(uuid.uuid4())
                        
                    # 查找对应的用户问题
                    user_query = ""
                    if i > 0 and st.session_state.messages[i-1]["role"] == "user":
                        user_query = st.session_state.messages[i-1]["content"]
                        
                    # 检查是否已经提供过反馈
                    feedback_key = f"{msg['message_id']}"
                    feedback_type_key = f"feedback_type_{feedback_key}"
                    
                    # 创建一个容器用于显示反馈结果
                    feedback_container = st.empty()
                    
                    if feedback_key not in st.session_state.feedback_given:
                        # 添加反馈按钮
                        col1, col2, col3 = st.columns([0.1, 0.1, 0.8])
                        
                        with col1:
                            thumbs_up_key = f"thumbs_up_{msg['message_id']}_{i}"
                            if st.button("👍", key=thumbs_up_key):
                                # 检查是否有正在处理的请求
                                if "feedback_in_progress" not in st.session_state:
                                    st.session_state.feedback_in_progress = False
                                
                                if st.session_state.feedback_in_progress:
                                    with feedback_container:
                                        st.warning("请等待当前操作完成...")
                                else:
                                    st.session_state.feedback_in_progress = True
                                    try:
                                        with feedback_container:
                                            with st.spinner("正在提交反馈..."):
                                                response = send_feedback(
                                                    msg["message_id"], 
                                                    user_query, 
                                                    True, 
                                                    st.session_state.session_id,
                                                    st.session_state.agent_type
                                                )
                                        
                                        st.session_state.feedback_given.add(feedback_key)
                                        st.session_state[feedback_type_key] = "positive"
                                        
                                        # 根据响应显示不同的消息
                                        with feedback_container:
                                            if response and "action" in response:
                                                if "高质量" in response["action"]:
                                                    st.success("感谢您的肯定！此回答已被标记为高质量。", icon="🙂")
                                                else:
                                                    st.success("感谢您的反馈！", icon="👍")
                                            else:
                                                st.info("已收到您的反馈。", icon="ℹ️")
                                    except Exception as e:
                                        st.error(f"提交反馈时出错: {str(e)}")
                                    finally:                                           
                                        st.session_state.feedback_in_progress = False
                                    
                        with col2:
                            thumbs_down_key = f"thumbs_down_{msg['message_id']}_{i}"
                            if st.button("👎", key=thumbs_down_key):
                                # 检查是否有正在处理的请求
                                if "feedback_in_progress" not in st.session_state:
                                    st.session_state.feedback_in_progress = False
                                
                                if st.session_state.feedback_in_progress:
                                    with feedback_container:
                                        st.warning("请等待当前操作完成...")
                                else:
                                    st.session_state.feedback_in_progress = True
                                    try:
                                        with feedback_container:
                                            with st.spinner("正在提交反馈..."):
                                                response = send_feedback(
                                                    msg["message_id"], 
                                                    user_query, 
                                                    False, 
                                                    st.session_state.session_id,
                                                    st.session_state.agent_type
                                                )
                                        
                                        st.session_state.feedback_given.add(feedback_key)
                                        st.session_state[feedback_type_key] = "negative"
                                        
                                        # 根据响应显示不同的消息
                                        with feedback_container:
                                            if response and "action" in response:
                                                if "清除" in response["action"]:
                                                    st.error("已收到您的反馈，此回答将不再使用。", icon="🔄")
                                                else:
                                                    st.error("已收到您的反馈，我们会改进。", icon="👎")
                                            else:
                                                st.info("已收到您的反馈。", icon="ℹ️")
                                    except Exception as e:
                                        st.error(f"提交反馈时出错: {str(e)}")
                                    finally:
                                        st.session_state.feedback_in_progress = False
                    else:
                        # 显示已提供的反馈类型
                        feedback_type = st.session_state.get(feedback_type_key, None)
                        with feedback_container:
                            if feedback_type == "positive":
                                st.success("您已对此回答给予肯定！", icon="👍")
                            elif feedback_type == "negative":
                                st.error("您已对此回答提出改进建议。", icon="👎")
                            else:
                                st.info("已收到您的反馈。", icon="ℹ️")
                
                    # 如果是AI回答且有源内容引用，显示查看源内容按钮
                    if st.session_state.debug_mode and st.session_state.agent_type != "deep_research_agent":
                        source_ids = extract_source_ids(msg["content"])
                        if source_ids:
                            with st.expander("查看引用源文本", expanded=False):
                                # 获取所有源文件信息
                                source_infos = get_source_file_info_batch(source_ids)
                                
                                for s_idx, source_id in enumerate(source_ids):
                                    # 使用缓存的信息
                                    display_name = source_infos.get(source_id, {}).get("file_name", f"源文本 {source_id}")
                                    source_btn_key = f"src_{source_id}_{i}_{s_idx}"
                                    
                                    if st.button(f"加载 {display_name}", key=source_btn_key):
                                        with st.spinner(f"加载源文本 {display_name}..."):
                                            source_data = get_source_content(source_id)
                                            if source_data and "content" in source_data:
                                                st.session_state.source_content = source_data["content"]
                                                st.session_state.current_tab = "源内容"
                                                st.rerun()
                        
                        # 如果是最后一条AI消息，添加自动提取图谱按钮 - deep_research_agent禁用此功能
                        if st.session_state.agent_type != "deep_research_agent":
                            extract_kg_key = f"extract_kg_{i}"
                            if st.button("提取知识图谱", key=extract_kg_key):
                                with st.spinner("提取知识图谱数据..."):
                                    # 获取对应的用户查询
                                    user_query = ""
                                    if i > 0 and st.session_state.messages[i-1]["role"] == "user":
                                        user_query = st.session_state.messages[i-1]["content"]
                                        
                                    # 使用用户查询来过滤知识图谱
                                    kg_data = get_knowledge_graph_from_message(msg["content"], user_query)
                                    if kg_data and len(kg_data.get("nodes", [])) > 0:
                                        # 确保当前消息有正确的kg_data
                                        st.session_state.messages[i]["kg_data"] = kg_data
                                        # 更新当前的图谱消息索引为当前处理的消息索引
                                        st.session_state.current_kg_message = i
                                        st.session_state.current_tab = "知识图谱"  # 自动切换到知识图谱标签
                                        st.rerun()
        
        # 处理新消息
        if prompt := st.chat_input("请输入您的问题...", key="chat_input"):
            # 检查是否有正在处理的请求
            if "processing_lock" not in st.session_state:
                st.session_state.processing_lock = False
                
            if st.session_state.processing_lock:
                st.warning("请等待当前操作完成...")
                return
                
            st.session_state.processing_lock = True
            
            with st.chat_message("user"):
                st.write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("assistant"):
                try:
                    with st.spinner("思考中..."):
                        response = send_message(prompt)
                    
                    if response:
                        answer_content = response["answer"]
                        
                        # 保存迭代信息和原始思考过程到会话状态
                        if "iterations" in response:
                            st.session_state.iterations = response["iterations"]
                        if "raw_thinking" in response:
                            st.session_state.raw_thinking = response["raw_thinking"]
                        
                        # 保存执行日志
                        if response.get("execution_log"):
                            st.session_state.execution_log = response["execution_log"]
                        
                        # 保存原始执行日志
                        if response.get("execution_logs"):
                            st.session_state.execution_logs = response["execution_logs"]
                        
                        # 创建消息对象
                        message_obj = {
                            "role": "assistant",
                            "content": answer_content,
                            "message_id": str(uuid.uuid4())
                        }
                        
                        # 处理deep_research_agent的思考过程
                        if st.session_state.agent_type == "deep_research_agent":
                            # 判断是否需要显示思考过程
                            show_thinking = st.session_state.get("show_thinking", False)
                            
                            # 优先使用raw_thinking
                            if "raw_thinking" in response:
                                raw_thinking = response["raw_thinking"]
                                message_obj["raw_thinking"] = raw_thinking
                                message_obj["processed_content"] = answer_content
                                
                                # 如果设置了显示思考过程
                                if show_thinking:
                                    # 直接显示思考过程，使用引用格式
                                    thinking_lines = raw_thinking.split('\n')
                                    quoted_thinking = '\n'.join([f"> {line}" for line in thinking_lines])
                                    st.markdown(quoted_thinking)
                                    
                                    # 添加两行空行间隔
                                    st.markdown("\n\n")
                                    
                                    # 显示最终答案
                                    st.markdown(answer_content)
                                else:
                                    # 不显示思考过程，直接显示答案
                                    st.markdown(answer_content)
                            # 检查answer中是否有<think>标签
                            elif "<think>" in answer_content and "</think>" in answer_content:
                                # 提取<think>标签中的内容
                                thinking_match = re.search(r'<think>(.*?)</think>', answer_content, re.DOTALL)
                                
                                if thinking_match:
                                    thinking_process = thinking_match.group(1)
                                    # 移除思考过程，保留答案
                                    processed_content = answer_content.replace(f"<think>{thinking_process}</think>", "").strip()
                                    
                                    # 保存处理后的内容和思考过程
                                    message_obj["raw_thinking"] = thinking_process
                                    message_obj["processed_content"] = processed_content
                                    
                                    # 如果设置了显示思考过程
                                    if show_thinking:
                                        # 格式化思考过程，使用引用格式
                                        thinking_lines = thinking_process.split('\n')
                                        quoted_thinking = '\n'.join([f"> {line}" for line in thinking_lines])
                                        
                                        # 显示思考过程
                                        st.markdown(quoted_thinking)
                                        
                                        # 添加两行空行间隔
                                        st.markdown("\n\n")
                                        
                                        # 显示答案
                                        st.markdown(processed_content)
                                    else:
                                        # 不显示思考过程，直接显示答案
                                        st.markdown(processed_content)
                                else:
                                    # 如果提取失败，显示完整内容但移除可能的<think>标签
                                    cleaned_content = re.sub(r'<think>|</think>', '', answer_content)
                                    st.markdown(cleaned_content)
                            else:
                                # 普通回答，无思考过程
                                st.markdown(answer_content)
                        else:
                            # 普通回答
                            st.markdown(answer_content)
                        
                        # 保存引用数据
                        if "reference" in response:
                            message_obj["reference"] = response["reference"]
                            
                        # 添加消息对象到会话
                        st.session_state.messages.append(message_obj)
                        
                        # 从回答中提取知识图谱数据，deep_research_agent禁用此功能
                        if st.session_state.debug_mode and st.session_state.agent_type != "deep_research_agent":
                            with st.spinner("提取知识图谱数据..."):
                                # 获取当前新消息的索引，即最后一条消息
                                current_msg_index = len(st.session_state.messages) - 1
                                
                                # 优先使用后端返回的kg_data
                                kg_data = response.get("kg_data")
                                
                                # 如果后端没有返回kg_data，尝试从回答中提取，并传递用户查询
                                if not kg_data or len(kg_data.get("nodes", [])) == 0:
                                    kg_data = get_knowledge_graph_from_message(response["answer"], prompt)
                                
                                if kg_data and len(kg_data.get("nodes", [])) > 0:
                                    # 更新该消息的kg_data
                                    st.session_state.messages[current_msg_index]["kg_data"] = kg_data
                                    
                                    # 更新当前处理的图谱消息索引为最新消息的索引
                                    st.session_state.current_kg_message = current_msg_index
                                    
                                    # 自动切换到知识图谱标签
                                    st.session_state.current_tab = "知识图谱"
                                    st.rerun()
                                else:
                                    st.error("无法提取知识图谱数据")
                except Exception as e:
                    st.error(f"处理消息时出错: {str(e)}")
                finally:
                    # 确保请求处理完成后释放锁
                    st.session_state.processing_lock = False
                    
            st.rerun()

def clear_chat_with_lock_reset():
    """清除聊天并重置处理锁"""
    # 重置处理锁
    st.session_state.processing_lock = False
    # 调用原始清除函数
    clear_chat()