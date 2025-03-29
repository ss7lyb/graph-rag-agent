可以通过--save_dir指定保存的评估结果的目录

**评估Graph Agent（带调试）：**

```bash
python evaluate_graph_agent.py --questions_file questions.json --verbose
```

**评估Hybrid Agent（带标准答案）：**

```bash
python evaluate_hybrid_agent.py --questions_file questions.json --golden_answers_file answer.json
```

**仅评估Naive Agent的答案质量：**

```bash
python evaluate_naive_agent.py --questions_file questions.json --golden_answers_file answer.json --eval_type answer
```

**评估Deep Research Agent（使用增强版工具）：**

```bash
python evaluate_deep_agent.py --questions_file questions.json --use_deeper
```

### 比较所有Agent

使用主脚本来比较所有Agent的性能：

```bash
python evaluate_all_agents.py --questions_file questions.json --golden_answers_file answer.json  --verbose
```

**仅比较部分Agent：**

```bash
python evaluate_all_agents.py --questions_file questions.json --agents graph,hybrid 
```

**仅比较检索性能：**

```bash
python evaluate_all_agents.py --questions_file questions.json --eval_type retrieval
```

**使用自定义指标：**

```bash
python evaluate_all_agents.py --questions_file questions.json --metrics em,f1,retrieval_precision
```

某次运行的结果：

| 指标 | naive | hybrid | graph | deep |
| --- | --- | --- | --- | --- |
| **答案质量指标** |  |  |  |  |
| answer_comprehensiveness | 0.9667 | 1.0000 | 0.6000 | 0.9333 |
| em | 0.6333 | 0.6000 | 0.4000 | 0.6000 |
| f1 | 0.6303 | 0.6667 | 0.4333 | 0.6230 |
| factual_consistency | 1.0000 | 0.8667 | 0.9667 | 0.9333 |
| response_coherence | 1.0000 | 1.0000 | 0.7000 | 1.0000 |
| **LLM评估指标** |  |  |  |  |
| Comprehensiveness | 0.8333 | 0.8667 | 0.6667 | 0.8333 |
| Directness | 0.9000 | 0.8000 | 0.8333 | 0.9333 |
| Empowerment | 0.7500 | 0.7333 | 0.5333 | 0.7333 |
| Relativeness | 0.9500 | 0.8333 | 0.8000 | 0.9667 |
| Total | 0.8550 | 0.8117 | 0.7000 | 0.8617 |
| **检索性能指标** |  |  |  |  |
| retrieval_latency | 9.3777 | 14.9450 | 6.3789 | 38.0844 |
| retrieval_precision | 0.4333 | 0.5000 | 0.4000 | 0.4000 |
| retrieval_utilization | 0.6000 | 0.5000 | 0.4000 | 0.4667 |