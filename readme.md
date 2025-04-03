# GraphRAG + DeepSearch 实现与 Agent 构建

本项目聚焦于结合 **GraphRAG** 与 **私域 Deep Search** 的方式，实现可解释、可推理的智能问答系统，同时结合多 Agent 协作与知识图谱增强，构建完整的 RAG 智能交互解决方案。

> 灵感来源于检索增强推理与深度搜索场景，探索 RAG 与 Agent 在未来应用中的结合路径。

## 相关资源

- [大模型推理能力不断增强，RAG 和 Agent 何去何从](https://www.bilibili.com/video/BV1i6RNYpEwV)  
- [企业级知识图谱交互问答系统方案](https://www.bilibili.com/video/BV1U599YrE26)  
- [Jean - 用国产大模型 + LangChain + Neo4j 建图全过程](https://zhuanlan.zhihu.com/p/716089164)
- [GraphRAG vs DeepSearch？GraphRAG 提出者给你答案](https://mp.weixin.qq.com/s/FOT4pkEPHJR8xFvcVk1YFQ)

![svg](./assets/deepsearch.svg)

## 项目亮点

- **从零开始复现 GraphRAG**：完整实现了 GraphRAG 的核心功能，将知识表示为图结构
- **DeepSearch 与 GraphRAG 创新融合**：现有 DeepSearch 框架主要基于向量数据库，本项目创新性地将其与知识图谱结合
- **多 Agent 协同架构**：实现不同类型 Agent 的协同工作，提升复杂问题处理能力
- **完整评估系统**：提供 20+ 种评估指标，全方位衡量系统性能
- **增量更新机制**：支持知识图谱的动态增量构建与智能去重
- **思考过程可视化**：展示 AI 的推理轨迹，提高可解释性和透明度

## 快速开始

请参考：[快速开始文档](./assets/start.md)

## 功能模块

### 图谱构建与管理

- **多格式文档处理**：支持 TXT、PDF、MD、DOCX、DOC、CSV、JSON、YAML/YML 等格式
- **LLM 驱动的实体关系提取**：利用大语言模型从文本中识别实体与关系
- **增量更新机制**：支持已有图谱上的动态更新，智能处理冲突
- **社区检测与摘要**：自动识别知识社区并生成摘要，支持 Leiden 和 SLLPA 算法
- **一致性验证**：内置图谱一致性检查与修复机制

### GraphRAG 实现

- **多级检索策略**：支持本地搜索、全局搜索、混合搜索等多种模式
- **图谱增强上下文**：利用图结构丰富检索内容，提供更全面的知识背景
- **Chain of Exploration**：实现在知识图谱上的多步探索能力
- **社区感知检索**：根据知识社区结构优化搜索结果

### DeepSearch 融合

- **多步骤思考-搜索-推理**：支持复杂问题的分解与深入挖掘
- **证据链追踪**：记录每个推理步骤的证据来源，提高可解释性
- **思考过程可视化**：实时展示 AI 的推理轨迹
- **多路径并行搜索**：同时执行多种搜索策略，综合利用不同知识来源

### 多种 Agent 实现

- **NaiveRagAgent**：基础向量检索型 Agent，适合简单问题
- **GraphAgent**：基于图结构的 Agent，支持关系推理
- **HybridAgent**：混合多种检索方式的 Agent
- **DeepResearchAgent**：深度研究型 Agent，支持复杂问题多步推理
- **FusionGraphRAGAgent**：融合型 Agent，结合多种策略的优势

### 系统评估与监控

- **多维度评估**：包括答案质量、检索性能、图评估和深度研究评估
- **性能监控**：跟踪 API 调用耗时，优化系统性能
- **用户反馈机制**：收集用户对回答的评价，持续改进系统

### 前后端实现

- **流式响应**：支持 AI 生成内容的实时流式显示
- **交互式知识图谱**：提供 Neo4j 风格的图谱交互界面
- **调试模式**：开发者可查看执行轨迹和搜索过程
- **RESTful API**：完善的后端 API 设计，支持扩展开发

## 未来规划

1. **自动化数据获取**：
   - 加入定时爬虫功能，替代当前的手动文档更新方式
   - 实现资源自动发现与增量爬取

2. **图谱构建优化**：
   - 采用 GRPO 训练小模型支持图谱抽取
   - 降低当前 DeepResearch 进行图谱抽取的成本与延迟

3. **领域特化嵌入**：
   - 解决语义相近但概念不同的术语区分问题
   - 优化如"优秀学生"vs"国家奖学金"、"过失杀人"vs"故意杀人"等的嵌入区分

4. **Agent 性能优化**：
   - 提升 Agent 框架响应速度
   - 优化多 Agent 协作机制

## 参考与致谢

- [GraphRAG](https://github.com/microsoft/graphrag) – 微软开源的知识图谱增强 RAG 框架  
- [llm-graph-builder](https://github.com/neo4j-labs/llm-graph-builder) – Neo4j 官方 LLM 建图工具  
- [LightRAG](https://github.com/HKUDS/LightRAG) – 轻量级知识增强生成方案  
- [deep-searcher](https://github.com/zilliztech/deep-searcher) – Zilliz团队开源的私域语义搜索框架  
- [ragflow](https://github.com/infiniflow/ragflow) – 企业级 RAG 系统