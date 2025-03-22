from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
import json
from models.schemas import ChatRequest, ChatResponse, ClearRequest, ClearResponse
from services.chat_service import process_chat, process_chat_stream
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

@router.post("/chat/stream")
async def chat_stream(request: Request):
    """流式响应聊天请求"""
    # 解析请求数据
    data = await request.json()
    message = data.get("message")
    session_id = data.get("session_id")
    
    # 设置流式响应
    async def event_generator():
        try:
            # 确保明确设置格式为SSE
            yield "data: " + json.dumps({"status": "start"}) + "\n\n"
            
            # 处理消息流
            async for chunk in process_chat_stream(
                message=message,
                session_id=session_id,
                # ...其他参数
            ):
                # 确保每个事件都正确格式化为SSE
                yield "data: " + chunk + "\n\n"
                
            # 发送完成事件
            yield "data: " + json.dumps({"status": "done"}) + "\n\n"
        except Exception as e:
            # 发送错误事件
            yield "data: " + json.dumps({"status": "error", "message": str(e)}) + "\n\n"
    
    # 返回流式响应
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # 阻止Nginx缓冲
        }
    )


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