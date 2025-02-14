import sys
from pathlib import Path
# 将项目根目录添加到 Python 路径
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from processor.file_reader import FileReader
from model.get_models import get_embeddings_model
from typing import List, Tuple

from config.settings import FILES_DIR

class VectorOperations:
    """
    向量运算类，用于处理embedding相关的计算和测试
    """
    def __init__(self, files_dir: str):
        """
        初始化向量运算类
        
        Args:
            files_dir: 文件目录路径
        """
        self.embeddings = get_embeddings_model()
        self.file_reader = FileReader(files_dir)

    def cosine_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            vector1: 第一个向量
            vector2: 第二个向量
            
        Returns:
            float: 余弦相似度值
            
        Raises:
            ValueError: 当输入向量为零向量时抛出
        """
        # 计算向量的点积
        dot_product = np.dot(vector1, vector2)
        
        # 计算向量的范数（长度）
        norm_vector1 = np.linalg.norm(vector1)
        norm_vector2 = np.linalg.norm(vector2)
        
        # 检查是否存在零向量
        if norm_vector1 == 0 or norm_vector2 == 0:
            raise ValueError("输入向量之一是零向量，无法计算余弦相似度。")
        
        # 计算余弦相似度
        return dot_product / (norm_vector1 * norm_vector2)

    def test_short_text_embeddings(self, texts: List[str]) -> List[Tuple[str, str, float]]:
        """
        测试短文本的embedding相似度
        
        Args:
            texts: 待测试的文本列表
            
        Returns:
            List[Tuple[str, str, float]]: 相邻文本对的相似度结果
        """
        results = self.embeddings.embed_documents(texts)
        similarities = []
        
        # 计算相邻文本对的相似度
        for i in range(0, len(results)-1, 2):
            similarity = self.cosine_similarity(results[i], results[i+1])
            similarities.append((texts[i], texts[i+1], similarity))
            
        return similarities

    def test_long_text_embeddings(self) -> List[float]:
        """
        测试长文本的embedding相似度
        
        Returns:
            List[float]: 不同文本片段间的相似度列表
        """
        # 读取文件内容
        file_contents = self.file_reader.read_txt_files()
        
        if not file_contents:
            print("没有找到文件内容")
            return []
            
        text = file_contents[0][1]  # 使用第一个文件的内容
        
        # 获取不同文本片段的embedding
        full_vector = self.embeddings.embed_query(text)
        partial_vector1 = self.embeddings.embed_query(text[100:4100])
        partial_vector2 = self.embeddings.embed_query(
            text[100:2000] + text[2100:4100]
        )
        
        # 计算相似度
        similarities = [
            self.cosine_similarity(full_vector, partial_vector1),
            self.cosine_similarity(full_vector, partial_vector2)
        ]
        
        return similarities

if __name__ == "__main__":

    vector_ops = VectorOperations(FILES_DIR)
    
    # 测试短文本
    test_texts = ["沙僧", "沙和尚", "猪", "猪八戒", "孙悟空", "悟空"]
    short_text_results = vector_ops.test_short_text_embeddings(test_texts)
    
    print("短文本相似度测试结果：")
    for text1, text2, similarity in short_text_results:
        print(f"{text1} 与 {text2} 的相似度: {similarity:.4f}")
    print()
    
    # 测试长文本
    long_text_results = vector_ops.test_long_text_embeddings()
    
    print("长文本相似度测试结果：")
    print(f"完整文本与部分文本1的相似度: {long_text_results[0]:.4f}")
    print(f"完整文本与部分文本2的相似度: {long_text_results[1]:.4f}")