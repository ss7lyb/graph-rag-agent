# 🚀 快速开始指南

## 🧩 One-API 部署

使用 Docker 启动 One-API：

```bash
docker run --name one-api -d --restart always \
  -p 13000:3000 \
  -e TZ=Asia/Shanghai \
  -v /home/ubuntu/data/one-api:/data \
  justsong/one-api
```

在 One-API 控制台中配置第三方 API Key。本项目的所有 API 请求将通过 One-API 转发。

---

## 🧠 Neo4j 启动

```bash
cd graph-rag-agent/
docker compose up -d
```

默认账号密码：

```
用户名：neo4j
密码：12345678
```

---

## 🛠️ 环境搭建

```bash
conda create -n graphrag python==3.10
conda activate graphrag
cd graph-rag-agent/
pip install -r requirements.txt
```

📎 注意：如需处理 `.doc` 格式（旧版 Word 文件），请根据操作系统安装相应依赖，详见 `requirements.txt` 中注释：

```txt
# Linux
sudo apt-get install python-dev-is-python3 libxml2-dev libxslt1-dev antiword unrtf poppler-utils

# Windows
pywin32>=302

textract==1.6.3  # Windows 无需安装
```

---

## ⚙️ .env 配置

在项目根目录下创建 `.env` 文件，示例如下：

```env
# One-API 配置
OPENAI_API_KEY='你的 one-api 令牌'
OPENAI_BASE_URL='http://localhost:13000/v1'

# 模型配置
OPENAI_EMBEDDINGS_MODEL='嵌入模型名称'
OPENAI_LLM_MODEL='LLM 模型名称'
TEMPERATURE=0
MAX_TOKENS=2000
VERBOSE=True

# Neo4j 配置
NEO4J_URI='neo4j://localhost:7687'
NEO4J_USERNAME='neo4j'
NEO4J_PASSWORD='12345678'

# LangSmith 配置（可选）
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="你的 LangSmith API Key"
LANGSMITH_PROJECT="项目名称"
```

---

## 🧱 项目初始化

```bash
pip install -e .
```

---

## 📂 知识图谱原始文件放置

请将原始文件放入 `files/` 文件夹。当前支持以下格式（采用简单分块，后续会优化处理方式）：

```
- TXT（纯文本）
- PDF（PDF 文档）
- MD（Markdown）
- DOCX（新版 Word 文档）
- DOC（旧版 Word 文档）
- CSV（表格）
- JSON（结构化文本）
- YAML/YML（配置文件）
```

---

## ⚙️ 知识图谱配置（`config/settings.py`）

```python
# 基础设置
theme = "悟空传"
entity_types = ["人物", "妖怪", "位置"]
relationship_types = ["师徒", "师兄弟", "对抗", "对话", "态度", "故事地点", "其它"]

# 图谱参数
similarity_threshold = 0.9
community_algorithm = 'leiden'  # 可选：sllpa 或 leiden

# 文本分块参数
CHUNK_SIZE = 300
OVERLAP = 50

# 回答方式
response_type = "多个段落"

# Agent 工具描述
lc_description = "用于需要具体细节的查询，例如《悟空传》中的对话、场景描写等。"
gl_description = "用于宏观总结和分析，如人物关系、主题发展等。"
naive_description = "基础检索工具，返回最相关的原文段落。"
```

---

## 🔧 构建知识图谱

```bash
cd graph-rag-agent/
python build/main.py
```

📌 **注意：** `main.py`是构建的全流程，如果需要单独跑某个流程，请先完成实体索引的构建，再进行 chunk 索引构建，否则会报错（chunk 索引依赖实体索引）。

---

## 🔍 知识图谱搜索测试

```bash
cd graph-rag-agent/test

# 非流式查询
python search_without_stream.py

# 流式查询
python search_with_stream.py
```

---

## 📊 知识图谱评估

```bash
cd evaluator/test
# 查看对应 README 获取更多信息
```

---

## 💬 示例问题配置（用于前端展示）

编辑 `config/settings.py` 中的 `examples` 字段：

```python
examples = [
    "《悟空传》的主要人物有哪些？",
    "唐僧和会说话的树讨论了什么？",
    "孙悟空跟女妖之间有什么故事？",
    "他最后的选择是什么？"
]
```

---

## 🧵 并发进程配置（`server/main.py`）

```python
# FastAPI 的并发进程数设置
workers = 2
```

---

## ⏱️ 深度搜索优化（建议禁用前端超时）

如需开启深度搜索功能，建议禁用前端超时限制，修改 `frontend/utils/api.py`：

```python
response = requests.post(
    f"{API_URL}/chat",
    json={
        "message": message,
        "session_id": st.session_state.session_id,
        "debug": st.session_state.debug_mode,
        "agent_type": st.session_state.agent_type
    },
    # timeout=120  # 建议注释掉此行
)
```

---

## 🎨 中文字体支持（Linux）

如需中文图表显示，可参考[字体安装教程](https://zhuanlan.zhihu.com/p/571610437)。默认使用英文绘图（`matplotlib`）。

---

## 🚀 启动前后端服务

```bash
# 启动后端
cd graph-rag-agent/
python server/main.py

# 启动前端
cd graph-rag-agent/
streamlit run frontend/app.py
```