import hashlib
import time
import concurrent.futures
from typing import List
from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document

from dotenv import load_dotenv

load_dotenv()

class GraphStructureBuilder:
    def __init__(self, batch_size=100):
        self.graph = Neo4jGraph(refresh_schema=True)
        self.batch_size = batch_size
            
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
        """创建Chunk节点并建立关系 - 批处理优化版本"""
        t0 = time.time()
        
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
            
            # 当累积了一定量的数据时，进行批处理
            if len(batch_data) >= self.batch_size:
                self._process_batch(file_name, batch_data, relationships)
                batch_data = []
                relationships = []
        
        # 处理剩余的数据
        if batch_data:
            self._process_batch(file_name, batch_data, relationships)
        
        t1 = time.time()
        
        return lst_chunks_including_hash
    
    def _process_batch(self, file_name: str, batch_data: List[dict], relationships: List[dict]):
        """批量处理一组chunks和关系"""
        if not batch_data:
            return
            
        # 分离FIRST_CHUNK和NEXT_CHUNK关系
        first_relationships = [r for r in relationships if r.get("type") == "FIRST_CHUNK"]
        next_relationships = [r for r in relationships if r.get("type") == "NEXT_CHUNK"]
        
        # 使用优化的数据库操作
        self._create_chunks_and_relationships_optimized(file_name, batch_data, first_relationships, next_relationships)
    
    def _create_chunks_and_relationships_optimized(self, file_name: str, batch_data: List[dict], 
                                                  first_relationships: List[dict], next_relationships: List[dict]):
        """优化的创建chunks和关系的查询 - 减少数据库往返"""
        # 合并查询：创建Chunk节点和PART_OF关系
        query_chunks_and_part_of = """
        UNWIND $batch_data AS data
        MERGE (c:`__Chunk__` {id: data.id})
        SET c.text = data.pg_content, 
            c.position = data.position, 
            c.length = data.length, 
            c.fileName = data.f_name,
            c.content_offset = data.content_offset, 
            c.tokens = data.tokens
        WITH c, data
        MATCH (d:`__Document__` {fileName: data.f_name})
        MERGE (c)-[:PART_OF]->(d)
        """
        self.graph.query(query_chunks_and_part_of, params={"batch_data": batch_data})
        
        # 处理FIRST_CHUNK关系
        if first_relationships:
            query_first_chunk = """
            UNWIND $relationships AS relationship
            MATCH (d:`__Document__` {fileName: $f_name})
            MATCH (c:`__Chunk__` {id: relationship.chunk_id})
            MERGE (d)-[:FIRST_CHUNK]->(c)
            """
            self.graph.query(query_first_chunk, params={
                "f_name": file_name,
                "relationships": first_relationships
            })
        
        # 处理NEXT_CHUNK关系
        if next_relationships:
            query_next_chunk = """
            UNWIND $relationships AS relationship
            MATCH (c:`__Chunk__` {id: relationship.current_chunk_id})
            MATCH (pc:`__Chunk__` {id: relationship.previous_chunk_id})
            MERGE (pc)-[:NEXT_CHUNK]->(c)
            """
            self.graph.query(query_next_chunk, params={"relationships": next_relationships})
    
    def bulk_create_relation_between_chunks(self, file_name: str, chunks: List) -> list:
        """一次性创建所有Chunk节点和关系 - 适用于较小数据集"""
        # 此方法类似于原始的create_relation_between_chunks，但使用一个合并的查询来处理所有数据
        
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
        
        # 合并查询：一次性完成所有节点和关系的创建
        query_combined = """
        // 第一步：创建所有Chunk节点
        UNWIND $batch_data AS data
        MERGE (c:`__Chunk__` {id: data.id})
        SET c.text = data.pg_content, 
            c.position = data.position, 
            c.length = data.length, 
            c.fileName = data.f_name,
            c.content_offset = data.content_offset, 
            c.tokens = data.tokens
        
        // 第二步：匹配文档并创建PART_OF关系
        WITH collect(c) as chunks
        MATCH (d:`__Document__` {fileName: $f_name})
        UNWIND chunks as c
        MERGE (c)-[:PART_OF]->(d)
        
        // 第三步：创建FIRST_CHUNK关系
        WITH d
        MATCH (first_chunk:`__Chunk__` {id: $first_chunk_id})
        MERGE (d)-[:FIRST_CHUNK]->(first_chunk)
        """
        
        # 获取第一个chunk的ID
        first_chunk_id = None
        if relationships and relationships[0].get("type") == "FIRST_CHUNK":
            first_chunk_id = relationships[0].get("chunk_id")
        
        # 如果有数据，执行合并查询
        if first_chunk_id and batch_data:
            self.graph.query(
                query_combined, 
                params={
                    "batch_data": batch_data,
                    "f_name": file_name,
                    "first_chunk_id": first_chunk_id
                }
            )
            
            # 单独处理NEXT_CHUNK关系
            query_next_chunk = """
            UNWIND $relationships AS relationship
            MATCH (c:`__Chunk__` {id: relationship.current_chunk_id})
            MATCH (pc:`__Chunk__` {id: relationship.previous_chunk_id})
            MERGE (pc)-[:NEXT_CHUNK]->(c)
            """
            
            next_relationships = [r for r in relationships if r.get("type") == "NEXT_CHUNK"]
            if next_relationships:
                self.graph.query(query_next_chunk, params={"relationships": next_relationships})
        
        return lst_chunks_including_hash
    
    def parallel_process_chunks(self, file_name: str, chunks: List, max_workers=4) -> list:
        """并行处理chunks，提高大量数据的处理速度"""
        if len(chunks) < 100:  # 对于小数据集，使用标准方法
            return self.create_relation_between_chunks(file_name, chunks)
        
        # 将chunks分为多个批次
        chunk_batches = []
        batch_size = max(10, len(chunks) // max_workers)
        
        for i in range(0, len(chunks), batch_size):
            chunk_batches.append(chunks[i:i+batch_size])
        
        # 为每个批次准备处理函数
        def process_chunk_batch(batch, start_index):
            results = []
            current_chunk_id = ""
            batch_data = []
            relationships = []
            offset = 0
            
            if start_index > 0 and start_index < len(chunks):
                # 获取前一个chunk的ID作为起始点
                prev_chunk = chunks[start_index - 1]
                prev_content = ''.join(prev_chunk)
                current_chunk_id = hashlib.sha1(prev_content.encode()).hexdigest()
                # 计算前面所有chunk的offset
                for j in range(start_index):
                    offset += len(''.join(chunks[j]))
            
            # 处理批次内的每个chunk
            for i, chunk in enumerate(batch):
                abs_index = start_index + i
                page_content = ''.join(chunk)
                page_content_sha1 = hashlib.sha1(page_content.encode())
                previous_chunk_id = current_chunk_id
                current_chunk_id = page_content_sha1.hexdigest()
                position = abs_index + 1
                
                if i > 0:
                    last_page_content = ''.join(batch[i-1])
                    offset += len(last_page_content)
                    
                firstChunk = (abs_index == 0)
                
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
                
                results.append({
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
            
            return {
                "batch_data": batch_data,
                "relationships": relationships,
                "results": results
            }
        
        # 并行处理所有批次
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(process_chunk_batch, batch, i * batch_size): i
                for i, batch in enumerate(chunk_batches)
            }
            
            # 收集所有处理结果
            all_batch_data = []
            all_relationships = []
            all_results = []
            
            for future in concurrent.futures.as_completed(future_to_batch):
                result = future.result()
                all_batch_data.extend(result["batch_data"])
                all_relationships.extend(result["relationships"])
                all_results.extend(result["results"])
        
        # 写入数据库
        self._create_chunks_and_relationships(file_name, all_batch_data, all_relationships)
        
        return all_results
    
    def _create_chunks_and_relationships(self, file_name: str, batch_data: List[dict], relationships: List[dict]):
        """执行创建chunks和关系的查询 - 保留原始方法"""
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