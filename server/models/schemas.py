from pydantic import BaseModel
from typing import Optional, List, Dict


class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str
    session_id: str
    debug: bool = False
    agent_type: str = "graph_agent"  # 默认采用 graphrag
    show_thinking: bool = False  # 是否显示思考过程，仅适用于deep_research_agent


class ChatResponse(BaseModel):
    """聊天响应模型"""
    answer: str
    execution_log: Optional[List[Dict]] = None
    kg_data: Optional[Dict] = None
    reference: Optional[Dict] = None
    iterations: Optional[List[Dict]] = None


class SourceRequest(BaseModel):
    """源内容请求模型"""
    source_id: str


class SourceResponse(BaseModel):
    """源内容响应模型"""
    content: str


class SourceInfoResponse(BaseModel):
    """源文件信息响应模型"""
    file_name: str


class ClearRequest(BaseModel):
    """清除聊天历史请求模型"""
    session_id: str


class ClearResponse(BaseModel):
    """清除聊天历史响应模型"""
    status: str
    remaining_messages: Optional[str] = None


class FeedbackRequest(BaseModel):
    """反馈请求模型"""
    message_id: str
    query: str
    is_positive: bool
    thread_id: str
    agent_type: Optional[str] = "graph_agent"


class FeedbackResponse(BaseModel):
    """反馈响应模型"""
    status: str
    action: str