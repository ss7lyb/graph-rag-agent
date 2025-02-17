import hanlp
from typing import List, Tuple
from config.settings import CHUNK_SIZE, OVERLAP, FILES_DIR
from processor.file_reader import FileReader

class ChineseTextChunker:
    """中文文本分块器，将长文本分割成带有重叠的文本块"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        """
        初始化分块器
        
        Args:
            chunk_size: 每个文本块的目标大小（tokens数量）
            overlap: 相邻文本块的重叠大小（tokens数量）
        """
        if chunk_size <= overlap:
            raise ValueError("chunk_size必须大于overlap")
            
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        
    def process_files(self, file_contents: List[Tuple[str, str]]) -> List[Tuple[str, str, List[List[str]]]]:
        """
        处理多个文件的内容
        
        Args:
            file_contents: List of (filename, content) tuples
            
        Returns:
            List of (filename, content, chunks) tuples
        """
        results = []
        for filename, content in file_contents:
            chunks = self.chunk_text(content)
            results.append((filename, content, chunks))
        return results
        
    def chunk_text(self, text: str) -> List[List[str]]:
        """
        将单个文本分割成块
        
        Args:
            text: 要分割的文本
            
        Returns:
            分割后的文本块列表，每个块是token列表
        """
        paragraphs = text.split('\n')
        chunks = []
        buffer = []
        
        i = 0
        while i < len(paragraphs):
            # 填充buffer
            while len(buffer) < self.chunk_size and i < len(paragraphs):
                tokens = self.tokenizer(paragraphs[i])
                buffer.extend(tokens)
                i += 1
                
            # 处理当前buffer
            while len(buffer) >= self.chunk_size:
                end = self._find_next_sentence_end(buffer, self.chunk_size)
                chunk = buffer[:end]
                chunks.append(chunk)
                
                start_next = self._find_previous_sentence_end(buffer, end - self.overlap)
                if start_next == 0:
                    start_next = self._find_previous_sentence_end(buffer, end - 1)
                if start_next == 0:
                    start_next = end - self.overlap
                    
                buffer = buffer[start_next:]
        
        # 处理剩余内容
        if buffer:
            last_chunk = chunks[-1]
            rest = ''.join(buffer)
            temp = ''.join(last_chunk[len(last_chunk)-len(rest):])
            if temp != rest:
                chunks.append(buffer)
                
        return chunks
    
    def _is_sentence_end(self, token: str) -> bool:
        """判断token是否为句子结束符"""
        return token in ['。', '！', '？']
    
    def _find_next_sentence_end(self, tokens: List[str], start_pos: int) -> int:
        """从指定位置向后查找句子结束位置"""
        for i in range(start_pos, len(tokens)):
            if self._is_sentence_end(tokens[i]):
                return i + 1
        return len(tokens)
    
    def _find_previous_sentence_end(self, tokens: List[str], start_pos: int) -> int:
        """从指定位置向前查找句子结束位置"""
        for i in range(start_pos - 1, -1, -1):
            if self._is_sentence_end(tokens[i]):
                return i + 1
        return 0
    

if __name__ == '__main__':
    # 读取文件
    fr = FileReader(FILES_DIR)
    file_contents = fr.read_txt_files()
    
    # 创建分块器实例
    chunker = ChineseTextChunker(CHUNK_SIZE, OVERLAP)
    
    # 处理每个文件并添加chunks到file_contents
    for file_content in file_contents:
        print("文件名:", file_content[0])
        chunks = chunker.chunk_text(file_content[1])
        file_content.append(chunks)
    
    # 打印分块结果
    for file_content in file_contents:
        print(f"File: {file_content[0]} Chunks: {len(file_content[2])}")
        for i, chunk in enumerate(file_content[2]):
            print(f"Chunk {i+1}: {len(chunk)} tokens.")
    
    # 打印第一个文件的第一个分块内容
    if file_contents:
        print(''.join(file_contents[0][2][0]))