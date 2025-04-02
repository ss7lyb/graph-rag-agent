import asyncio
import re
from agent.deep_research_agent import DeepResearchAgent
from agent.naive_rag_agent import NaiveRagAgent
from agent.graph_agent import GraphAgent
from agent.hybrid_agent import HybridAgent
from agent.fusion_agent import FusionGraphRAGAgent

# DeepResearchAgent 流式示例
async def test_deep_research_agent_stream():
    print("\n======= DeepResearchAgent 流式输出 =======")
    agent = DeepResearchAgent()
    
    # 不显示思考过程的流式输出
    print("\n --- 不显示思考过程 ---")
    async for chunk in agent.ask_stream("优秀学生要如何申请", show_thinking=False):
        # 处理字典类型的最终答案
        if isinstance(chunk, dict) and "answer" in chunk:
            print("\n[最终答案]")
            print(chunk["answer"])
        else:
            # 输出文本块
            print(chunk, end="", flush=True)
    
    print("\n\n --- 显示思考过程 ---")
    # 显示思考过程的流式输出
    async for chunk in agent.ask_stream("优秀学生要如何申请", show_thinking=True):
        if isinstance(chunk, dict) and "answer" in chunk:
            print("\n[最终答案]")
            print(chunk["answer"])
        else:
            print(chunk, end="", flush=True)
    
    # 切换到标准版研究工具并测试
    agent.is_deeper_tool(False)
    print("\n\n --- 标准版研究工具 ---")
    async for chunk in agent.ask_stream("优秀学生要如何申请"):
        if isinstance(chunk, dict) and "answer" in chunk:
            print("\n[最终答案]")
            print(chunk["answer"])
        else:
            print(chunk, end="", flush=True)

# DeeperResearchTool 流式示例
async def test_deeper_research_tool_stream():
    print("\n======= DeeperResearchTool 流式输出 =======")
    from search.tool.deeper_research_tool import DeeperResearchTool
    
    deeper_tool = DeeperResearchTool()
    
    # 使用thinking_stream方法
    print("\n --- thinking_stream方法 ---")
    async for chunk in deeper_tool.thinking_stream("优秀学生要如何申请"):
        if isinstance(chunk, dict) and "answer" in chunk:
            print("\n[最终答案]")
            print(chunk["answer"])
        else:
            print(chunk, end="", flush=True)
    
    # 使用search_stream方法
    print("\n\n --- search_stream方法 ---")
    async for chunk in deeper_tool.search_stream("优秀学生要如何申请"):
        if isinstance(chunk, dict) and "answer" in chunk:
            print("\n[最终答案]")
            print(chunk["answer"])
        else:
            print(chunk, end="", flush=True)

# GraphAgent 流式示例
async def test_graph_agent_stream():
    print("\n======= GraphAgent 流式输出 =======")
    agent = GraphAgent()
    
    # 使用ask_stream方法
    print("\n --- ask_stream方法 ---")
    async for chunk in agent.ask_stream("优秀学生要如何申请"):
        print(chunk, end="", flush=True)

# HybridAgent 流式示例
async def test_hybrid_agent_stream():
    print("\n======= HybridAgent 流式输出 =======")
    agent = HybridAgent()
    
    # 使用ask_stream方法
    print("\n --- ask_stream方法 ---")
    async for chunk in agent.ask_stream("优秀学生要如何申请"):
        print(chunk, end="", flush=True)

# NaiveRagAgent 流式示例
async def test_naive_agent_stream():
    print("\n======= NaiveRagAgent 流式输出 =======")
    agent = NaiveRagAgent()
    
    # 使用ask_stream方法
    print("\n --- ask_stream方法 ---")
    async for chunk in agent.ask_stream("优秀学生要如何申请"):
        print(chunk, end="", flush=True)

async def test_fusion_agent_stream():
    print("\n======= FusionGraphRagAgent 流式输出 =======")
    agent = FusionGraphRAGAgent()

    # 使用ask_stream方法
    print("\n --- ask_stream方法 ---")
    async for chunk in agent.ask_stream("优秀学生要如何申请"):
        print(chunk, end="", flush=True)

# 运行所有流式测试
async def run_all_stream_tests():
    await test_deep_research_agent_stream()
    # await test_deeper_research_tool_stream()
    await test_graph_agent_stream()
    await test_hybrid_agent_stream()
    await test_naive_agent_stream()
    await test_fusion_agent_stream()

# 运行主函数
if __name__ == "__main__":
    asyncio.run(run_all_stream_tests())