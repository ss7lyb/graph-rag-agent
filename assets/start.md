# 快速开始

one-api下载：

```bash
docker run --name one-api -d --restart always -p 13000:3000 -e TZ=Asia/Shanghai -v /home/ubuntu/data/one-api:/data justsong/one-api
```

之后在one-api中配置第三方api-key即可，本项目的api全部走one-api中转

neo4j下载：

```bash
cd graph-rag-agent/
docker compose up -d

# 用户名：neo4j
# 密码：12345678
```

环境构建：

```bash
conda create -n graphrag python==3.10
cd graph-rag-agent/
pip install -r requirements.txt
```

.env文件配置：

```env
OPENAI_API_KEY = '本地one-api的令牌'
OPENAI_BASE_URL = 'http://localhost:13000/v1'

OPENAI_EMBEDDINGS_MODEL = '嵌入模型名称'
OPENAI_LLM_MODEL = 'LLM模型名称'

TEMPERATURE = 0  # 温度
MAX_TOKENS = 2000  # 最大Token

VERBOSE = True # 是否打开调试模式

# neo4j配置
NEO4J_URI='neo4j://localhost:7687'
NEO4J_USERNAME='neo4j'
NEO4J_PASSWORD='12345678'

# langsmith配置
LANGSMITH_TRACING=true  # 是否启用
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="api-key"
LANGSMITH_PROJECT="项目名称"
```

项目初始化：

```bash
pip install -e .
```

知识图谱生成配置：config/settings.py:

```python
# 知识图谱设置
theme="悟空传"
entity_types=["人物","妖怪","位置"]
relationship_types=["师徒", "师兄弟", "对抗", "对话", "态度", "故事地点", "其它"]

# 实体相似度
similarity_threshold = 0.9

# 社区算法：sllpa or leiden
community_algorithm = 'leiden'

 文本分块
CHUNK_SIZE=300
OVERLAP=50

# 回答方式
response_type="多个段落"

# agent 工具描述
lc_description = "用于需要具体细节的查询。检索《悟空传》特定章节中的具体情节、对话、场景描写等详细内容。适用于'某个场景发生了什么'、'具体描写是怎样的'等问题。"
gl_description = "用于需要总结归纳的查询。分析《悟空传》小说的整体脉络、人物关系、主题发展等宏观内容。适用于'整个故事的发展'、'人物关系如何'等需要跨章节分析的问题。"
naive_description = "基础检索工具，直接查找与问题最相关的文本片段，不做复杂分析。快速获取《悟空传》中的原文内容，返回最匹配的原文段落。"
```

构建完整的知识图谱：

```bash
cd graph-rag-agent/
python build/main.py
```

注意，对chunk的索引构建，一定需要在entity的索引构建完之后才能进行，否则会报错。因为chunk的索引是依赖与entity里建立的索引构建的。

测试对知识图谱的搜索：

```bash
cd graph-rag-agent/
python search_test.py
```

前端的示例问题修改：config/settings.py:

```python
# 项目前端的“示例问题”显示
examples = [
    "《悟空传》的主要人物有哪些？",
    "唐僧和会说话的树讨论了什么？",
    "孙悟空跟女妖之间有什么故事？",
    "他最后的选择是什么？"
]
```

项目并发进程数：config/settings.py：

```python
# fastapi 并发进程数
workers = 2
```

前后端项目启动：

```bash
cd graph-rag-agent/
python server/main.py

cd graph-rag-agent/
streamlit run frontend/app.py
```