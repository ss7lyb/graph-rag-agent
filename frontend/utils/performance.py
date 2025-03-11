import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def display_performance_stats():
    """显示性能统计信息"""
    if 'performance_metrics' not in st.session_state or not st.session_state.performance_metrics:
        st.info("尚无性能数据")
        return
    
    # 计算消息响应时间统计
    message_times = [m["duration"] for m in st.session_state.performance_metrics 
                    if m["operation"] == "send_message"]
    
    if message_times:
        avg_time = sum(message_times) / len(message_times)
        max_time = max(message_times)
        min_time = min(message_times)
        
        st.subheader("消息响应性能")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("平均响应时间", f"{avg_time:.2f}s")
        with col2:
            st.metric("最大响应时间", f"{max_time:.2f}s")
        with col3:
            st.metric("最小响应时间", f"{min_time:.2f}s")
        
        # 绘制响应时间图表
        if len(message_times) > 1:
            fig, ax = plt.subplots(figsize=(10, 4))
            x = np.arange(len(message_times))
            ax.plot(x, message_times, marker='o')
            ax.set_title('Response Time Trend')
            ax.set_xlabel('Message ID')
            ax.set_ylabel('Response Time (s)')
            ax.grid(True)
            
            st.pyplot(fig)
    
    # 反馈性能统计
    feedback_times = [m["duration"] for m in st.session_state.performance_metrics 
                     if m["operation"] == "send_feedback"]
    
    if feedback_times:
        avg_feedback_time = sum(feedback_times) / len(feedback_times)
        st.subheader("反馈处理性能")
        st.metric("平均反馈处理时间", f"{avg_feedback_time:.2f}s")

def clear_performance_data():
    """清除性能数据"""
    if 'performance_metrics' in st.session_state:
        st.session_state.performance_metrics = []
        return True
    return False