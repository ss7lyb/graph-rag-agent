import logging
import re
from typing import List
from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship

class GraphWriter:
    def __init__(self, graph: Neo4jGraph):
        self.graph = graph
        
    def convert_to_graph_document(self, chunk_id: str, input_text: str, result: str) -> GraphDocument:
        """将提取的实体关系文本转换为GraphDocument对象"""
        # 编译正则表达式模式
        node_pattern = re.compile(r'\("entity" : "(.+?)" : "(.+?)" : "(.+?)"\)')
        relationship_pattern = re.compile(
            r'\("relationship" : "(.+?)" : "(.+?)" : "(.+?)" : "(.+?)" : (.+?)\)'
        )

        nodes = {}
        relationships = []

        # 解析节点
        for match in node_pattern.findall(result):
            node_id, node_type, description = match
            if node_id not in nodes:
                nodes[node_id] = Node(
                    id=node_id,
                    type=node_type,
                    properties={'description': description}
                )

        # 解析关系
        for match in relationship_pattern.findall(result):
            source_id, target_id, type, description, weight = match
            # 确保源节点存在
            if source_id not in nodes:
                nodes[source_id] = Node(
                    id=source_id,
                    type="未知",
                    properties={'description': 'No additional data'}
                )
            # 确保目标节点存在
            if target_id not in nodes:
                nodes[target_id] = Node(
                    id=target_id,
                    type="未知",
                    properties={'description': 'No additional data'}
                )
                
            # 创建关系
            relationships.append(
                Relationship(
                    source=nodes[source_id],
                    target=nodes[target_id],
                    type=type,
                    properties={
                        "description": description,
                        "weight": float(weight)
                    }
                )
            )

        # 创建并返回GraphDocument对象
        return GraphDocument(
            nodes=list(nodes.values()),
            relationships=relationships,
            source=Document(
                page_content=input_text,
                metadata={"chunk_id": chunk_id}
            )
        )
        
    def process_and_write_graph_documents(self, file_contents: List) -> None:
        """处理并写入所有文件的GraphDocument对象"""
        for file_content in file_contents:
            chunks = file_content[3]  # chunks_with_hash在索引3的位置
            results = file_content[4]  # 提取结果在索引4的位置
            
            # 创建GraphDocument对象
            graph_documents = []
            for chunk, result in zip(chunks, results):
                graph_document = self.convert_to_graph_document(
                    chunk["chunk_id"],
                    chunk["chunk_doc"].page_content,
                    result
                )
                graph_documents.append(graph_document)
                
            # 过滤掉空的图对象
            valid_graph_documents = [
                doc for doc in graph_documents
                if len(doc.nodes) > 0 or len(doc.relationships) > 0
            ]
            
            # 将有效的图文档写入数据库
            if valid_graph_documents:
                self.graph.add_graph_documents(
                    valid_graph_documents,
                    baseEntityLabel=True,
                    include_source=True
                )
                
            # 收集chunk_ids用于后续处理
            chunk_ids = [chunk["chunk_id"] for chunk in chunks]
            self.merge_chunk_relationships(chunk_ids)
    
    def merge_chunk_relationships(self, chunk_ids: List[str]) -> None:
        """合并Chunk节点与Document节点的关系"""
        logging.info("Creating MENTIONS relationships between chunks and entities")
        
        if not chunk_ids:
            return
            
        batch_data = [{"chunk_id": chunk_id} for chunk_id in chunk_ids]
        
        merge_query = """
            UNWIND $batch_data AS data
            MATCH (c:`__Chunk__` {id: data.chunk_id}), (d:Document{chunk_id:data.chunk_id})
            WITH c, d
            MATCH (d)-[r:MENTIONS]->(e)
            MERGE (c)-[newR:MENTIONS]->(e)
            ON CREATE SET newR += properties(r)
            DETACH DELETE d
        """
        
        self.graph.query(merge_query, params={"batch_data": batch_data})