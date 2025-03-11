from fastapi import APIRouter
from models.schemas import SourceRequest, SourceResponse
from services.kg_service import get_source_content

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