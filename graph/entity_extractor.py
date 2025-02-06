import time
from typing import List, Tuple
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

class EntityRelationExtractor:
    def __init__(self, llm, system_template, human_template, 
                 entity_types: List[str], relationship_types: List[str]):
        self.llm = llm
        self.entity_types = entity_types
        self.relationship_types = relationship_types
        self.chat_history = []
        
        # 设置分隔符
        self.tuple_delimiter = " : "
        self.record_delimiter = "\n"
        self.completion_delimiter = "\n\n"
        
        # 创建提示模板
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        
        self.chat_prompt = ChatPromptTemplate.from_messages([
            system_message_prompt,
            MessagesPlaceholder("chat_history"),
            human_message_prompt
        ])
        
        # 创建处理链
        self.chain = self.chat_prompt | self.llm
        
    def process_chunks(self, file_contents: List[Tuple]) -> List[Tuple]:
        """处理所有文件的所有chunks"""
        t0 = time.time()
        for file_content in file_contents:
            results = []
            for chunk in file_content[2]:
                t1 = time.time()
                
                # 处理单个chunk
                input_text = ''.join(chunk)
                result = self._process_single_chunk(input_text)
                results.append(result)
                
                # 打印处理信息
                t2 = time.time()
                print(f"Chunk processing time: {t2-t1} seconds")
                print(f"Input text: {input_text}\n")
                print(f"Result: {result}\n")
                
            print(f"File processing time: {t2-t0} seconds\n\n")
            file_content.append(results)
            
        return file_contents
    
    def _process_single_chunk(self, input_text: str) -> str:
        """处理单个文本块"""
        response = self.chain.invoke({
            "chat_history": self.chat_history,
            "entity_types": self.entity_types,
            "relationship_types": self.relationship_types,
            "tuple_delimiter": self.tuple_delimiter,
            "record_delimiter": self.record_delimiter,
            "completion_delimiter": self.completion_delimiter,
            "input_text": input_text
        })
        return response.content