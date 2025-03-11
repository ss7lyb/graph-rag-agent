import re
from typing import List
import streamlit as st

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