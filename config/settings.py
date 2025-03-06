from pathlib import Path

# 知识图谱设置
theme="悟空传"
entity_types=["人物","妖怪","位置"]
relationship_types=["师徒", "师兄弟", "对抗", "对话", "态度", "故事地点", "其它"]

# 实体相似度
similarity_threshold = 0.9

# 社区算法：sllpa or leiden
community_algorithm = 'leiden'

# 构建路径
BASE_DIR = Path(__file__).resolve().parent.parent
FILES_DIR = BASE_DIR / 'files'

# 文本分块
CHUNK_SIZE=300
OVERLAP=50

# 回答方式
response_type="多个段落"

# agent 工具描述
lc_description = "用于需要具体细节的查询。检索《悟空传》特定章节中的具体情节、对话、场景描写等详细内容。适用于'某个场景发生了什么'、'具体描写是怎样的'等问题。"
gl_description = "用于需要总结归纳的查询。分析《悟空传》小说的整体脉络、人物关系、主题发展等宏观内容。适用于'整个故事的发展'、'人物关系如何'等需要跨章节分析的问题。"

# 项目前端的“示例问题”显示
examples = [
    "《悟空传》的主要人物有哪些？",
    "唐僧和会说话的树讨论了什么？",
    "孙悟空跟女妖之间有什么故事？",
    "他最后的选择是什么？"
]