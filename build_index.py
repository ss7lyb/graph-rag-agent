import numpy as np

from model.get_models import get_embeddings_model
from processor.file_reader import FileReader
from config.settings import FILES_DIR

# 测试 embedding -----------------------------------------------
# 计算两个向量的余弦相似度
def cosine_similarity(vector1, vector2):
    # 计算向量的点积
    dot_product = np.dot(vector1, vector2)
    # 计算向量的范数（长度）
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    
    # 检查是否存在零向量，避免除以零
    if norm_vector1 == 0 or norm_vector2 == 0:
        raise ValueError("输入向量之一是零向量，无法计算余弦相似度。")
    
    # 计算余弦相似度
    cosine_sim = dot_product / (norm_vector1 * norm_vector2)
    return cosine_sim

embeddings = get_embeddings_model()

# 测试embedding的使用
tests=["沙僧","沙和尚","猪","猪八戒","孙悟空","悟空"]

results = embeddings.embed_documents(tests)

print(cosine_similarity(results[2],results[3]))
print('\n')

# 长文本Embedding
fr = FileReader(FILES_DIR)
file_contents = fr.read_txt_files()
print(len(file_contents[0][1]))
print('\n')

single_vector = embeddings.embed_query(file_contents[0][1])
temp = file_contents[0][1][100:4100]
single_vector2 = embeddings.embed_query(file_contents[0][1][100:4100])
print(cosine_similarity(single_vector,single_vector2))
print('\n')
single_vector3 = embeddings.embed_query(file_contents[0][1][100:2000]+file_contents[0][1][2100:4100])
print(cosine_similarity(single_vector,single_vector3))
print('\n')

# 实体索引 --------------------------------------------