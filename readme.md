# GraphRAG 实现与 Agent 构建

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