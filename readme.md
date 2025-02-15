# GraphRAG Agent构建

构建完整的知识图谱：

```bash
python build_graph_and_index.py 
```

执行对知识图谱的搜索：

```bash
python search_test.py
```

## 备注

本项目采用 Claude 进行模块拆分，即 `restruct` 的 git 记录，原版代码请参见：

- [build_graph](https://github.com/1517005260/graph-rag-agent/blob/8de857a36022eaf48ec3882e99e8c9de469be155/extract_entity.py)
- [build_index](https://github.com/1517005260/graph-rag-agent/blob/a0193a09f8918afb507ac65701ef05541665f67b/build_index.py)
- [search](https://github.com/1517005260/graph-rag-agent/blob/f746b3d7f8ef38c451ce291d25a32bd0cfc6e6b2/search.py)