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

agent测试：

```bash
# local search 与 global search
python agent/graph_agent.py 

# 混合搜索
python agent/hybrid_agent.py
```

前后端项目启动：

```bash
uvicorn backend:app --reload

streamlit run frontend.py
```