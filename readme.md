# GraphRAG + DeepSearch 实现与 Agent 构建

本项目聚焦于结合 **GraphRAG** 与 **私域 Deep Search** 的方式，实现可解释、可推理的智能问答系统，同时结合多 Agent 协作与知识图谱增强，构建完整的 RAG 智能交互解决方案。

> 灵感来源于检索增强推理与深度搜索场景，探索 RAG 与 Agent 在未来应用中的结合路径。


## 灵感与背景

**视频与文章推荐：**

- [大模型推理能力不断增强，RAG 和 Agent 何去何从](https://www.bilibili.com/video/BV1i6RNYpEwV)  
- [企业级知识图谱交互问答系统方案](https://www.bilibili.com/video/BV1U599YrE26)  
- [Jean - 用国产大模型 + LangChain + Neo4j 建图全过程](https://zhuanlan.zhihu.com/p/716089164)
- [GraphRAG vs DeepSearch？GraphRAG 提出者给你答案](https://mp.weixin.qq.com/s/FOT4pkEPHJR8xFvcVk1YFQ)

![svg](./assets/deepsearch.svg)


## 快速开始

请参考：[快速开始文档](./assets/start.md)


## 功能模块

1. **图谱构建**  
   从多格式文档中提取实体关系，自动构建 Neo4j 知识图谱。

2. **增量更新**
   
   支持已有图谱基础上的动态增量构建与去重更新。

3. **GraphRAG**  
   利用知识图谱增强上下文的生成式问答能力。

4. **私域 DeepSearch**  
   实现深度语义搜索 + 推理路径回溯，提升问答可解释性。

5. **GraphRAG + DeepSearch 融合**  
   将知识图谱增强的上下文与深度搜索的推理路径相结合，提供更精准且易于追溯的生成式答案。

6. **评估机制**  
   提供多维度自动评估 RAG 效果的能力，包括精确率、召回率、推理路径评估、生成质量评分等。

7. **前后端联动与流式输出**  
   使用 Streamlit 前端与 FastAPI 后端框架，实现前后端实时交互，支持答案流式返回与多 Agent 策略动态选择。

8. **图谱推理问答**  
   基于知识图谱实现结构化推理算法，包括实体间最短路径查询、共同邻居检索、多跳关系追踪等高级图查询功能。

9. **图谱增删改查接口**  
   内置完善的图谱编辑 API 接口，支持知识图谱动态增删改查及实时维护。


## 参考与致谢

- [GraphRAG](https://github.com/microsoft/graphrag) – 微软开源的知识图谱增强 RAG 框架  
- [llm-graph-builder](https://github.com/neo4j-labs/llm-graph-builder) – Neo4j 官方 LLM 建图工具  
- [LightRAG](https://github.com/HKUDS/LightRAG) – 轻量级知识增强生成方案  
- [deep-searcher](https://github.com/zilliztech/deep-searcher) – 硅基流动团队开源的私域语义搜索框架  
- [ragflow](https://github.com/infiniflow/ragflow) – 开源的企业级 RAG 系统