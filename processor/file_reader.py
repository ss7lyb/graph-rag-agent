import codecs
import os
from typing import List, Tuple

import sys
from pathlib import Path
# 将项目根目录添加到 Python 路径
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import FILES_DIR

class FileReader:
    def __init__(self, directory_path: str):
        self.directory_path = directory_path
        
    def read_txt_files(self):
        results = []
        for filename in os.listdir(self.directory_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.directory_path, filename)
                with codecs.open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                results.append([filename, content])
        return results
    
if __name__=='__main__':
    print(f"FILES_DIR: {FILES_DIR}")
    fr = FileReader(FILES_DIR)
    file_contents = fr.read_txt_files()
    for file_name, content in file_contents:
        print("文件名:", file_name)