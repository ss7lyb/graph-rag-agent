# GraphRAG 实现与 Agent 构建

## 快速开始

one-api下载：

```bash
docker run --name one-api -d --restart always -p 13000:3000 -e TZ=Asia/Shanghai -v /home/ubuntu/data/one-api:/data justsong/one-api
```

neo4j下载：

```bash
docker compose up -d

# 用户名：neo4j
# 密码：12345678
```

环境构建：

```bash
conda create -n graphrag python==3.10
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
```

构建完整的知识图谱：

```bash
python build/main.py
```

测试对知识图谱的搜索：

```bash
python search_test.py
```

前后端项目启动：

```bash
uvicorn backend:app --reload

streamlit run frontend.py
```