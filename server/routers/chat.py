from fastapi import APIRouter
from models.schemas import ChatRequest, ChatResponse, ClearRequest, ClearResponse
from services.chat_service import process_chat
from services.agent_service import agent_manager, format_execution_log
from utils.performance import measure_performance

# 创建路由器
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
@measure_performance("chat")
async def chat(request: ChatRequest):
    """
    处理聊天请求
    
    Args:
        request: 聊天请求
        
    Returns:
        ChatResponse: 聊天响应
    """
    result = await process_chat(
        message=request.message,
        session_id=request.session_id,
        debug=request.debug,
        agent_type=request.agent_type,
        use_deeper_tool=request.use_deeper_tool,
        show_thinking=request.show_thinking
    )
    
    if request.debug and "execution_log" in result:
        # 格式化执行日志
        result["execution_log"] = format_execution_log(result["execution_log"])
    
    return ChatResponse(**result)


@router.post("/clear", response_model=ClearResponse)
async def clear_chat(request: ClearRequest):
    """
    清除聊天历史
    
    Args:
        request: 清除请求
        
    Returns:
        ClearResponse: 清除响应
    """
    result = agent_manager.clear_history(request.session_id)
    return ClearResponse(**result)