from search.local_search import LocalSearch
from search.global_search import GlobalSearch
from model.get_models import get_llm_model, get_embeddings_model

llm = get_llm_model()
embeddings = get_embeddings_model()

# 普通使用
lc = LocalSearch(llm, embeddings)
result = lc.search("孙悟空对如来佛祖是一种什么样的心态？")
print(result)

print()

gl = GlobalSearch(llm)
result = gl.search("孙悟空对如来佛祖是一种什么样的心态？", level=0)
print(result)

# 使用上下文进行资源管理
with LocalSearch(llm, embeddings) as lc:
    result = lc.search("孙悟空对如来佛祖是一种什么样的心态？")
    print(result)

print()

with GlobalSearch(llm) as gl:
    result = gl.search("孙悟空对如来佛祖是一种什么样的心态？", level=0)
    print(result)