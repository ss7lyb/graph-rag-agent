import hashlib
import logging
from typing import List
from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document

from dotenv import load_dotenv

load_dotenv('../.env')

class GraphStructureBuilder:
    def __init__(self):
        self.graph = Neo4jGraph(refresh_schema=True)
        
    def clear_database(self):
        """清空数据库"""
        clear_query = """
            MATCH (n)
            DETACH DELETE n
            """
        self.graph.query(clear_query)
        
    def create_document(self, type: str, uri: str, file_name: str, domain: str):
        """创建Document节点"""
        query = """
        MERGE(d:`__Document__` {fileName: $file_name}) 
        SET d.type=$type, d.uri=$uri, d.domain=$domain
        RETURN d;
        """
        doc = self.graph.query(
            query,
            {"file_name": file_name, "type": type, "uri": uri, "domain": domain}
        )
        return doc
        
    def create_relation_between_chunks(self, file_name: str, chunks: List) -> list:
        """创建Chunk节点并建立关系"""
        logging.info("creating FIRST_CHUNK and NEXT_CHUNK relationships between chunks")
        current_chunk_id = ""
        lst_chunks_including_hash = []
        batch_data = []
        relationships = []
        offset = 0
        
        # 处理每个chunk
        for i, chunk in enumerate(chunks):
            page_content = ''.join(chunk)
            page_content_sha1 = hashlib.sha1(page_content.encode())
            previous_chunk_id = current_chunk_id
            current_chunk_id = page_content_sha1.hexdigest()
            position = i + 1
            
            if i > 0:
                last_page_content = ''.join(chunks[i-1])
                offset += len(last_page_content)
                
            firstChunk = (i == 0)
            
            # 创建metadata和Document对象
            metadata = {
                "position": position,
                "length": len(page_content),
                "content_offset": offset,
                "tokens": len(chunk)
            }
            chunk_document = Document(page_content=page_content, metadata=metadata)
            
            # 准备batch数据
            chunk_data = {
                "id": current_chunk_id,
                "pg_content": chunk_document.page_content,
                "position": position,
                "length": chunk_document.metadata["length"],
                "f_name": file_name,
                "previous_id": previous_chunk_id,
                "content_offset": offset,
                "tokens": len(chunk)
            }
            batch_data.append(chunk_data)
            
            lst_chunks_including_hash.append({
                'chunk_id': current_chunk_id,
                'chunk_doc': chunk_document
            })
            
            # 创建关系数据
            if firstChunk:
                relationships.append({"type": "FIRST_CHUNK", "chunk_id": current_chunk_id})
            else:
                relationships.append({
                    "type": "NEXT_CHUNK",
                    "previous_chunk_id": previous_chunk_id,
                    "current_chunk_id": current_chunk_id
                })
        
        # 执行创建查询
        self._create_chunks_and_relationships(file_name, batch_data, relationships)
        
        return lst_chunks_including_hash
    
    def _create_chunks_and_relationships(self, file_name: str, batch_data: List[dict], relationships: List[dict]):
        """执行创建chunks和关系的查询"""
        # 创建Chunk节点和PART_OF关系
        query_chunk_part_of = """
            UNWIND $batch_data AS data
            MERGE (c:`__Chunk__` {id: data.id})
            SET c.text = data.pg_content, 
                c.position = data.position, 
                c.length = data.length, 
                c.fileName = data.f_name,
                c.content_offset = data.content_offset, 
                c.tokens = data.tokens
            WITH data, c
            MATCH (d:`__Document__` {fileName: data.f_name})
            MERGE (c)-[:PART_OF]->(d)
        """
        self.graph.query(query_chunk_part_of, params={"batch_data": batch_data})
        
        # 创建FIRST_CHUNK关系
        query_first_chunk = """
            UNWIND $relationships AS relationship
            MATCH (d:`__Document__` {fileName: $f_name})
            MATCH (c:`__Chunk__` {id: relationship.chunk_id})
            FOREACH(r IN CASE WHEN relationship.type = 'FIRST_CHUNK' THEN [1] ELSE [] END |
                    MERGE (d)-[:FIRST_CHUNK]->(c))
        """
        self.graph.query(query_first_chunk, params={
            "f_name": file_name,
            "relationships": relationships
        })
        
        # 创建NEXT_CHUNK关系
        query_next_chunk = """
            UNWIND $relationships AS relationship
            MATCH (c:`__Chunk__` {id: relationship.current_chunk_id})
            WITH c, relationship
            MATCH (pc:`__Chunk__` {id: relationship.previous_chunk_id})
            FOREACH(r IN CASE WHEN relationship.type = 'NEXT_CHUNK' THEN [1] ELSE [] END |
                    MERGE (c)<-[:NEXT_CHUNK]-(pc))
        """
        self.graph.query(query_next_chunk, params={"relationships": relationships})