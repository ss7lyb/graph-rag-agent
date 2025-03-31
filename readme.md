# GraphRAG + DeepSearch 实现与 Agent 构建

本项目聚焦于结合 **GraphRAG** 与 **私域 Deep Search** 的方式，实现可解释、可推理的智能问答系统，同时结合多 Agent 协作与知识图谱增强，构建完整的 RAG 智能交互解决方案。

> 灵感来源于多模态检索增强推理与深度搜索场景，探索 RAG 与 Agent 在未来应用中的结合路径。

---

## 🔗 灵感与背景

📽️ **视频与文章推荐：**

- 🎥 [大模型推理能力不断增强，RAG 和 Agent 何去何从](https://www.bilibili.com/video/BV1i6RNYpEwV)  
- 🎥 [企业级知识图谱交互问答系统方案](https://www.bilibili.com/video/BV1U599YrE26)  
- ✍️ [Jean - 用国产大模型 + LangChain + Neo4j 建图全过程](https://zhuanlan.zhihu.com/p/716089164)

![svg](./assets/deepsearch.svg)

---

## 🚀 快速开始

请参考：[快速开始文档 👉](./assets/start.md)

---

## 🎯 功能模块

1. ✅ **图谱构建**  
   从多格式文档中提取实体关系，自动构建 Neo4j 知识图谱。

2. 🔁 **增量更新**  （正在开发）
   支持已有图谱基础上的动态增量构建与去重更新。

3. 🧩 **GraphRAG**  
   利用知识图谱增强上下文的生成式问答能力。

4. 🔍 **私域 DeepSearch**  
   实现深度语义搜索 + 推理路径回溯，提升问答可解释性。

5. 📈 **评估机制**  
   多维度自动评估 RAG 效果（精确率、召回率、推理路径等）。

6. 🖥️ **前后端联动与流式输出**  
   Streamlit 前端 + FastAPI 后端，支持流式返回与多 Agent 策略选择。

7. 🧠 **图谱推理问答**  
   支持基于知识图谱的结构化图算法推理问答，如实体间最短路径查询、共同邻居检索、多跳关系追踪等。

8. ✏️ **图谱增删改查**  
   内置图谱编辑接口，支持图谱动态维护。

---

## 🙏 参考与致谢

- [GraphRAG](https://github.com/microsoft/graphrag) – 微软开源的知识图谱增强 RAG 框架  
- [llm-graph-builder](https://github.com/neo4j-labs/llm-graph-builder) – Neo4j 官方 LLM 建图工具  
- [LightRAG](https://github.com/HKUDS/LightRAG) – 轻量级知识增强生成方案  
- [deep-searcher](https://github.com/zilliztech/deep-searcher) – Milvus 团队开源的私域语义搜索框架  
- [ragflow](https://github.com/infiniflow/ragflow) – 支持自动管道构建的 RAG 系统  