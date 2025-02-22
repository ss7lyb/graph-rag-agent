import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
from neo4j import GraphDatabase, Result
from dotenv import load_dotenv

load_dotenv()

import shutup
shutup.please()

from langchain_core.messages import RemoveMessage, AIMessage, HumanMessage, ToolMessage
from agent.graph_agent import GraphAgent

app = FastAPI()
agent = GraphAgent()
driver = GraphDatabase.driver(os.getenv('NEO4J_URI'), auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD')))

# Pydantic 模型定义
class ChatRequest(BaseModel):
    message: str
    session_id: str
    debug: bool = False

class ChatResponse(BaseModel):
    answer: str
    execution_log: Optional[List[Dict]] = None

class SourceRequest(BaseModel):
    source_id: str

class SourceResponse(BaseModel):
    content: str

class ClearRequest(BaseModel):
    session_id: str

class ClearResponse(BaseModel):
    status: str
    remaining_messages: Optional[str] = None

def format_messages_for_response(messages: List[Dict]) -> str:
    """将消息格式化为字符串"""
    formatted = []
    for msg in messages:
        if isinstance(msg, (HumanMessage, AIMessage)):
            prefix = "User: " if isinstance(msg, HumanMessage) else "AI: "
            formatted.append(f"{prefix}{msg.content}")
    return "\\n".join(formatted)

def format_execution_log(log: List[Dict]) -> List[Dict]:
    """格式化执行日志用于 JSON 响应"""
    formatted_log = []
    for entry in log:
        if isinstance(entry["input"], dict):
            input_str = {}
            for k, v in entry["input"].items():
                if isinstance(v, str):
                    input_str[k] = v
                else:
                    input_str[k] = str(v)
        else:
            input_str = str(entry["input"])
            
        if isinstance(entry["output"], dict):
            output_str = {}
            for k, v in entry["output"].items():
                if isinstance(v, str):
                    output_str[k] = v
                else:
                    output_str[k] = str(v)
        else:
            output_str = str(entry["output"])

        formatted_entry = {
            "node": entry["node"],
            "input": input_str,
            "output": output_str
        }
        formatted_log.append(formatted_entry)
    return formatted_log

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """处理聊天请求"""
    try:
        if request.debug:
            result = agent.ask_with_trace(
                request.message, 
                thread_id=request.session_id
            )
            return ChatResponse(
                answer=result["answer"],
                execution_log=format_execution_log(result["execution_log"])
            )
        else:
            answer = agent.ask(
                request.message, 
                thread_id=request.session_id
            )
            return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear", response_model=ClearResponse)
async def clear_chat(request: ClearRequest):
    """清除聊天历史"""
    try:
        config = {"configurable": {"thread_id": request.session_id}}
        
        messages = agent.memory.get(config)["channel_values"]["messages"]

        i = len(messages)
        for message in reversed(messages):
            if isinstance(messages[2], ToolMessage) and i == 4:
                break
            agent.graph.update_state(config, {"messages": RemoveMessage(id=message.id)})
            i = i - 1
            if i == 2:
                break

        remaining_messages = agent.memory.get(config)["channel_values"]["messages"]
        remaining_text = ""
        for msg in remaining_messages:
            if isinstance(msg, (AIMessage, HumanMessage)):
                prefix = "AI: " if isinstance(msg, AIMessage) else "User: "
                remaining_text += f"{prefix}{msg.content}\n"
        
        return ClearResponse(
            status="success",
            remaining_messages=remaining_text
        )
            
    except Exception as e:
        return ClearResponse(
            status="error",
            remaining_messages=f"未预期的错误: {str(e)}"
        )

@app.post("/source", response_model=SourceResponse)
async def get_source(request: SourceRequest):
    """处理源内容请求"""
    try:
        source_id = request.source_id
        id_parts = source_id.split(",")
        
        if id_parts[0] == "2":  # 文本块查询
            query = """
            MATCH (n:__Chunk__) 
            WHERE n.id = $id 
            RETURN n.fileName, n.text
            """
            params = {"id": id_parts[-1]}
        else:  # 社区查询
            query = """
            MATCH (n:__Community__) 
            WHERE n.id = $id 
            RETURN n.summary, n.full_content
            """
            params = {"id": id_parts[1]}

        result = driver.execute_query(
            query,
            parameters_=params,
            result_transformer_=Result.to_df
        )

        if result.shape[0] > 0:
            if id_parts[0] == "2":
                content = f"文件名: {result.iloc[0,0]}\n\n{result.iloc[0,1]}"
            else:
                content = f"摘要:\n{result.iloc[0,0]}\n\n全文:\n{result.iloc[0,1]}"
        else:
            content = "未找到相关内容"
            
        return SourceResponse(content=content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)