# from search.local_search import LocalSearch
# from search.global_search import GlobalSearch
# from model.get_models import get_llm_model, get_embeddings_model

# llm = get_llm_model()
# embeddings = get_embeddings_model()

# question = "孙悟空跟女妖之间有什么故事？"

# # 普通使用
# lc = LocalSearch(llm, embeddings)
# result = lc.search(question)
# print(result)

# print()

# gl = GlobalSearch(llm)
# result = gl.search(question, level=0)
# print(result)

# # 使用上下文进行资源管理
# with LocalSearch(llm, embeddings) as lc:
#     result = lc.search(question)
#     print(result)

# print()

# with GlobalSearch(llm) as gl:
#     result = gl.search(question, level=0)
#     print(result)

# 私域deepresearch
from agent.deep_research_agent import DeepResearchAgent

agent = DeepResearchAgent()
result = agent.ask_with_thinking("我旷课了30学时，我会被退学吗？")
print("\n ===============")
print(result)

print("\n ===============")
print(result.get('answer'))

# 切换到标准版研究工具
agent.is_deeper_tool(False)

standard_answer = agent.ask_with_thinking("我旷课了30学时，我会被退学吗？")
print("\n ===============")
print(standard_answer)