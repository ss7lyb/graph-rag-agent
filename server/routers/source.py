from fastapi import APIRouter
from models.schemas import SourceRequest, SourceResponse, SourceInfoResponse
from services.kg_service import get_source_content, get_source_file_info

# 创建路由器
router = APIRouter()


@router.post("/source", response_model=SourceResponse)
async def source(request: SourceRequest):
    """
    处理源内容请求
    
    Args:
        request: 源内容请求
        
    Returns:
        SourceResponse: 源内容响应
    """
    content = get_source_content(request.source_id)
    return SourceResponse(content=content)

@router.post("/source_info")
async def source_info(request: SourceRequest):
    """
    处理源文件信息请求
    
    Args:
        request: 源内容请求
        
    Returns:
        Dict: 包含文件名等信息的响应
    """
    info = get_source_file_info(request.source_id)
    return info