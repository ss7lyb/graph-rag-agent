# 社区检测与摘要模块

## 概述

本模块为图数据库中的社区检测与摘要功能提供支持，作为大型知识图谱项目的组成部分。主要功能包括识别图数据中的社区结构，并为每个社区生成摘要描述，便于用户理解社区的主要内容和特征。

## 功能特性

### 社区检测功能

- 支持两种社区检测算法：
  - **Leiden算法**：适用于大规模图数据的社区检测
  - **SLLPA算法**：标签传播算法的改进版本
- 自适应系统资源调整：根据可用内存和CPU自动调整算法参数
- 图投影优化：包含多种投影方法，适应不同规模的图数据
- 故障恢复机制：当标准处理方法失败时，提供备用处理路径

### 社区摘要功能

- 基于LLM生成社区内容的语义摘要
- 社区排名：根据社区重要性计算排名
- 并行处理：支持多线程并行生成社区摘要
- 批量处理：对大规模社区数据进行分批处理

## 技术架构

### 依赖组件

- **Neo4j图数据库**：用于存储和查询图数据
- **Graph Data Science库（GDS）**：提供图算法支持
- **LangChain**：用于构建LLM应用流程
- **psutil**：系统资源监控

### 设计模式

- **工厂模式**：用于创建不同类型的检测器和摘要生成器
- **混入类（Mixin）**：提供图投影等共享功能
- **上下文管理器**：管理资源生命周期
- **模板方法模式**：在基类中定义算法骨架，子类实现具体步骤

## 使用方法

### 社区检测

```python
from langchain_community.graphs import Neo4jGraph
from graphdatascience import GraphDataScience
from community import CommunityDetectorFactory

# 初始化图连接
graph = Neo4jGraph(url="neo4j://localhost:7687", username="neo4j", password="password")
gds = GraphDataScience("bolt://localhost:7687", auth=("neo4j", "password"))

# 创建社区检测器（可选算法：'leiden'或'sllpa'）
detector = CommunityDetectorFactory.create('leiden', gds, graph)

# 执行社区检测
results = detector.process()
print(f"社区检测结果: {results}")
```

### 社区摘要生成

```python
from community import CommunitySummarizerFactory

# 根据使用的检测算法创建对应的摘要生成器
summarizer = CommunitySummarizerFactory.create_summarizer('leiden', graph)

# 生成社区摘要
summaries = summarizer.process_communities()
print(f"已生成 {len(summaries)} 个社区摘要")
```

## 性能考量

- 对于大规模图（>50,000节点），社区检测过程可能耗时较长
- 内存使用量与图大小成正比，大图分析时需要足够的系统资源
- 社区摘要生成依赖LLM性能，可能受到API限制影响

## 错误处理

模块内置多层错误处理机制：
1. 首先尝试最优配置执行
2. 遇到问题时降级到更保守的配置
3. 提供详细的错误日志和执行统计信息

## 扩展性

- 可通过继承`BaseCommunityDetector`添加新的社区检测算法
- 可通过继承`BaseSummarizer`实现自定义摘要生成逻辑