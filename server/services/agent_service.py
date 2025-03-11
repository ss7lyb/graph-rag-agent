from typing import Dict, List
from langchain_core.messages import RemoveMessage, AIMessage, HumanMessage, ToolMessage


# 创建Agent管理类
class AgentManager:
    """Agent管理类"""
    
    def __init__(self):
        """初始化Agent管理器"""
        # 导入各种Agent
        from agent.graph_agent import GraphAgent
        from agent.hybrid_agent import HybridAgent
        from agent.naive_rag_agent import NaiveRagAgent
        
        # 创建Agent字典
        self.agents = {
            "graph_agent": GraphAgent(),
            "hybrid_agent": HybridAgent(),
            "naive_rag_agent": NaiveRagAgent(),
        }
    
    def get_agent(self, agent_type: str):
        """
        获取指定类型的Agent
        
        Args:
            agent_type: Agent类型名称
            
        Returns:
            Agent实例
        """
        if agent_type not in self.agents:
            raise ValueError(f"未知的agent类型: {agent_type}")
        return self.agents[agent_type]
    
    def clear_history(self, session_id: str) -> Dict:
        """
        清除特定会话的聊天历史
        
        Args:
            session_id: 会话ID
            
        Returns:
            Dict: 清除结果信息
        """
        remaining_text = ""
        
        try:
            # 清除所有agent的历史
            for agent_name, agent in self.agents.items():
                config = {"configurable": {"thread_id": session_id}}
                
                # 添加检查，防止None值报错
                memory_content = agent.memory.get(config)
                if memory_content is None or "channel_values" not in memory_content:
                    continue  # 跳过这个agent
                    
                messages = memory_content["channel_values"]["messages"]
                
                # 如果消息少于2条，不进行删除操作
                if len(messages) <= 2:
                    continue

                i = len(messages)
                for message in reversed(messages):
                    if isinstance(messages[2], ToolMessage) and i == 4:
                        break
                    agent.graph.update_state(config, {"messages": RemoveMessage(id=message.id)})
                    i = i - 1
                    if i == 2:  # 保留前两条消息
                        break

            # 获取剩余消息
            try:
                # 使用graph_agent检查剩余消息
                graph_agent = self.agents["graph_agent"]
                memory_content = graph_agent.memory.get({"configurable": {"thread_id": session_id}})
                
                if memory_content and "channel_values" in memory_content:
                    remaining_messages = memory_content["channel_values"]["messages"]
                    for msg in remaining_messages:
                        if isinstance(msg, (AIMessage, HumanMessage)):
                            prefix = "AI: " if isinstance(msg, AIMessage) else "User: "
                            remaining_text += f"{prefix}{msg.content}\n"
            except Exception as e:
                print(f"获取剩余消息失败: {e}")
        
        except Exception as e:
            print(f"清除聊天历史时出错: {str(e)}")
        
        return {
            "status": "success",
            "remaining_messages": remaining_text
        }
    
    def close_all(self):
        """关闭所有Agent资源"""
        for agent_name, agent in self.agents.items():
            try:
                agent.close()
                print(f"已关闭 {agent_name} 资源")
            except Exception as e:
                print(f"关闭 {agent_name} 资源时出错: {e}")


# 创建全局实例
agent_manager = AgentManager()


def format_messages_for_response(messages: List[Dict]) -> str:
    """
    将消息格式化为字符串
    
    Args:
        messages: 消息列表
    
    Returns:
        str: 格式化后的消息字符串
    """
    formatted = []
    for msg in messages:
        if isinstance(msg, (HumanMessage, AIMessage)):
            prefix = "User: " if isinstance(msg, HumanMessage) else "AI: "
            formatted.append(f"{prefix}{msg.content}")
    return "\n".join(formatted)


def format_execution_log(log: List[Dict]) -> List[Dict]:
    """
    格式化执行日志用于JSON响应
    
    Args:
        log: 原始执行日志
    
    Returns:
        List[Dict]: 格式化后的执行日志
    """
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