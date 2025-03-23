from agent.deep_research_agent import DeepResearchAgent
from agent.naive_rag_agent import NaiveRagAgent
from agent.graph_agent import GraphAgent
from agent.hybrid_agent import HybridAgent

# DeepResearchAgent 非流式示例
def test_deep_research_agent():
    print("\n======= DeepResearchAgent 非流式输出 =======")
    agent = DeepResearchAgent()
    
    # 使用ask_with_thinking方法获取带思考过程的结果
    result = agent.ask_with_thinking("优秀学生要如何申请")
    print("\n --- 完整结果 ---")
    print(result)
    
    print("\n --- 只有答案 ---")
    print(result.get('answer'))
    
    # 使用标准ask方法
    standard_answer = agent.ask("优秀学生要如何申请")
    print("\n --- 标准ask方法结果 ---")
    print(standard_answer)
    
    # 切换到标准版研究工具
    agent.is_deeper_tool(False)
    standard_result = agent.ask_with_thinking("优秀学生要如何申请")
    print("\n --- 标准版研究工具结果 ---")
    print(standard_result.get('answer'))

# DeeperResearchTool 非流式示例
def test_deeper_research_tool():
    print("\n======= DeeperResearchTool 非流式输出 =======")
    from search.tool.deeper_research_tool import DeeperResearchTool
    
    deeper_tool = DeeperResearchTool()
    
    # 使用thinking方法获取带思考过程的结果
    result = deeper_tool.thinking("优秀学生要如何申请")
    print("\n --- 完整结果 ---")
    print(result)
    
    print("\n --- 只有答案 ---")
    print(result.get('answer'))
    
    # 使用search方法获取简洁答案
    search_result = deeper_tool.search("优秀学生要如何申请")
    print("\n --- search方法结果 ---")
    print(search_result)

# GraphAgent 非流式示例
def test_graph_agent():
    print("\n======= GraphAgent 非流式输出 =======")
    agent = GraphAgent()
    
    # 使用ask方法
    result = agent.ask("优秀学生要如何申请")
    print("\n --- ask方法结果 ---")
    print(result)
    
    # 使用ask_with_trace获取执行轨迹
    trace_result = agent.ask_with_trace("优秀学生要如何申请")
    print("\n --- 带执行轨迹的结果 ---")
    print(trace_result.get("answer"))
    print("\n --- 执行轨迹 ---")
    for log in trace_result.get("execution_log", [])[:3]:  # 只显示前3条日志
        print(f"节点: {log.get('node')}, 时间: {log.get('timestamp')}")

# HybridAgent 非流式示例
def test_hybrid_agent():
    print("\n======= HybridAgent 非流式输出 =======")
    agent = HybridAgent()
    
    # 使用ask方法
    result = agent.ask("优秀学生要如何申请")
    print("\n --- ask方法结果 ---")
    print(result)
    
    # 使用ask_with_trace获取执行轨迹
    trace_result = agent.ask_with_trace("优秀学生要如何申请")
    print("\n --- 带执行轨迹的结果 ---")
    print(trace_result.get("answer"))

# NaiveRagAgent 非流式示例
def test_naive_agent():
    print("\n======= NaiveRagAgent 非流式输出 =======")
    agent = NaiveRagAgent()
    
    # 使用ask方法
    result = agent.ask("优秀学生要如何申请")
    print("\n --- ask方法结果 ---")
    print(result)
    
    # 使用ask_with_trace获取执行轨迹
    trace_result = agent.ask_with_trace("优秀学生要如何申请")
    print("\n --- 带执行轨迹的结果 ---")
    print(trace_result.get("answer"))

# 运行所有非流式测试
if __name__ == "__main__":
    test_deep_research_agent()
    test_deeper_research_tool()
    test_graph_agent()
    test_hybrid_agent()
    test_naive_agent()