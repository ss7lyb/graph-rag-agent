import time
import concurrent.futures
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Neo4jVector
from model.get_models import get_embeddings_model
from config.neo4jdb import get_db_manager

class ChunkIndexManager:
    """
    Chunk索引管理器，负责在Neo4j数据库中创建和管理文本块的向量索引。
    处理__Chunk__节点的embedding向量计算和索引创建，支持后续基于向量相似度的RAG查询。
    """
    
    def __init__(self, refresh_schema: bool = True, batch_size: int = 100, max_workers: int = 4):
        """
        初始化Chunk索引管理器
        
        Args:
            refresh_schema: 是否刷新Neo4j图数据库的schema
            batch_size: 批处理大小
            max_workers: 并行工作线程数
        """
        # 初始化图数据库连接
        db_manager = get_db_manager()
        self.graph = db_manager.graph
        
        # 初始化嵌入模型
        self.embeddings = get_embeddings_model()
        
        # 批处理和并行参数
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        # 性能监控参数
        self.embedding_time = 0
        self.db_time = 0
        
        # 创建必要的索引
        self._create_indexes()
    
    def _create_indexes(self) -> None:
        """创建必要的索引以优化查询性能"""
        index_queries = [
            "CREATE INDEX IF NOT EXISTS FOR (c:`__Chunk__`) ON (c.id)",
            "CREATE INDEX IF NOT EXISTS FOR (c:`__Chunk__`) ON (c.fileName)",
            "CREATE INDEX IF NOT EXISTS FOR (c:`__Chunk__`) ON (c.position)"
        ]
        
        for query in index_queries:
            self.graph.query(query)
        
    def clear_existing_index(self) -> None:
        """清除已存在的普通索引（不尝试删除向量索引）"""
        try:
            self.graph.query("DROP INDEX chunk_embedding IF EXISTS")
            print("已删除普通索引（如果存在）")
        except Exception as e:
            print(f"清除索引时出错 (可忽略): {e}")

    def create_chunk_index(self, 
                           node_label: str = '__Chunk__',
                           text_property: str = 'text',
                           embedding_property: str = 'embedding') -> Optional[Neo4jVector]:
        """
        为文本块节点生成embeddings并创建向量存储接口
        
        Args:
            node_label: 文本块节点的标签
            text_property: 用于计算embedding的文本属性
            embedding_property: 存储embedding的属性名
            
        Returns:
            Neo4jVector: 创建的向量存储对象
        """
        start_time = time.time()
        
        # 不尝试删除向量索引，只处理需要计算embedding的节点
        
        # 获取所有需要处理的文本块节点
        chunks = self.graph.query(
            f"""
            MATCH (c:`{node_label}`)
            WHERE c.{text_property} IS NOT NULL AND c.{embedding_property} IS NULL
            RETURN id(c) AS neo4j_id, c.id AS chunk_id
            """
        )
        
        if not chunks:
            print("没有找到需要处理的文本块节点")
            # 即使没有需要处理的节点，也尝试创建向量存储接口
            try:
                vector_store = Neo4jVector.from_existing_graph(
                    self.embeddings,
                    node_label=node_label,
                    text_node_properties=[text_property],
                    embedding_node_property=embedding_property
                )
                
                print("成功连接到现有向量索引")
                return vector_store
            except Exception as e:
                print(f"连接到向量存储时出错: {e}")
                return None
            
        print(f"开始为 {len(chunks)} 个文本块生成embeddings")
        
        # 批量处理所有文本块
        self._process_embeddings_in_batches(chunks, node_label, text_property, embedding_property)
        
        # 不尝试创建新的向量索引，只连接到现有的
        try:
            # 创建向量存储对象
            vector_store = Neo4jVector.from_existing_graph(
                self.embeddings,
                node_label=node_label,
                text_node_properties=[text_property],
                embedding_node_property=embedding_property
            )
            
            end_time = time.time()
            print(f"索引创建成功，总耗时: {end_time - start_time:.2f}秒")
            print(f"其中: embedding计算: {self.embedding_time:.2f}秒, 数据库操作: {self.db_time:.2f}秒")
            
            return vector_store
        except Exception as e:
            print(f"创建向量存储时出错: {e}")
            return None
    
    def _process_embeddings_in_batches(self, chunks: List[Dict[str, Any]], 
                                      node_label: str, text_property: str, 
                                      embedding_property: str) -> None:
        """
        批量处理文本块embedding的生成
        
        Args:
            chunks: 文本块列表
            node_label: 节点标签
            text_property: 文本属性
            embedding_property: embedding属性名
        """
        # 动态批处理大小
        chunk_count = len(chunks)
        optimal_batch_size = min(self.batch_size, max(20, chunk_count // 10))
        total_batches = (chunk_count + optimal_batch_size - 1) // optimal_batch_size
        
        print(f"使用批处理大小: {optimal_batch_size}, 共 {total_batches} 批")
        
        # 保存每个批次的处理时间
        batch_times = []
        
        for batch_index in range(total_batches):
            batch_start = time.time()
            
            start_idx = batch_index * optimal_batch_size
            end_idx = min(start_idx + optimal_batch_size, chunk_count)
            batch = chunks[start_idx:end_idx]
            
            # 获取批次内所有文本块的文本
            chunk_texts = self._get_chunk_texts_batch(batch, text_property)
            
            # 使用高效的嵌入批处理
            embedding_start = time.time()
            
            # 并行计算embeddings
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 预创建嵌入任务
                embedding_tasks = []
                for text in chunk_texts:
                    # 添加强健性处理，确保文本不为空
                    safe_text = text if text and text.strip() else "empty chunk"
                    embedding_tasks.append(safe_text)
                
                # 批量执行嵌入任务
                embeddings = []
                
                # 分析批处理的最佳大小
                embed_batch_size = min(32, len(embedding_tasks))
                
                for i in range(0, len(embedding_tasks), embed_batch_size):
                    sub_batch = embedding_tasks[i:i+embed_batch_size]
                    try:
                        # 尝试使用批量嵌入方法
                        if hasattr(self.embeddings, 'embed_documents'):
                            sub_batch_embeddings = self.embeddings.embed_documents(sub_batch)
                            embeddings.extend(sub_batch_embeddings)
                        else:
                            # 回退到单个嵌入
                            futures = [executor.submit(self.embeddings.embed_query, text) for text in sub_batch]
                            for future in concurrent.futures.as_completed(futures):
                                try:
                                    embeddings.append(future.result())
                                except Exception as e:
                                    print(f"嵌入计算失败: {e}")
                                    # 添加零向量作为备用
                                    if hasattr(self.embeddings, 'embedding_size'):
                                        embeddings.append([0.0] * self.embeddings.embedding_size)
                                    else:
                                        # 假设使用通用嵌入大小
                                        embeddings.append([0.0] * 1536)
                    except Exception as e:
                        print(f"批量嵌入处理失败: {e}")
                        # 尝试单个嵌入作为回退
                        for text in sub_batch:
                            try:
                                embeddings.append(self.embeddings.embed_query(text))
                            except Exception as e2:
                                print(f"单个嵌入计算失败: {e2}")
                                # 添加零向量作为备用
                                if hasattr(self.embeddings, 'embedding_size'):
                                    embeddings.append([0.0] * self.embeddings.embedding_size)
                                else:
                                    embeddings.append([0.0] * 1536)
            
            embedding_end = time.time()
            self.embedding_time += (embedding_end - embedding_start)
            
            # 批量更新数据库
            db_start = time.time()
            self._update_embeddings_batch(batch, embeddings, embedding_property)
            db_end = time.time()
            self.db_time += (db_end - db_start)
            
            batch_end = time.time()
            batch_time = batch_end - batch_start
            batch_times.append(batch_time)
            
            # 计算平均时间和剩余时间
            avg_time = sum(batch_times) / len(batch_times)
            remaining_batches = total_batches - (batch_index + 1)
            estimated_remaining = avg_time * remaining_batches
            
            print(f"已处理批次 {batch_index+1}/{total_batches}, "
                  f"批次耗时: {batch_time:.2f}秒, "
                  f"平均: {avg_time:.2f}秒/批, "
                  f"预计剩余: {estimated_remaining:.2f}秒")
    
    def _get_chunk_texts_batch(self, chunks: List[Dict[str, Any]], text_property: str) -> List[str]:
        """
        获取批量文本块的文本内容
        
        Args:
            chunks: 文本块列表
            text_property: 文本属性
            
        Returns:
            List[str]: 文本块文本列表
        """
        # 构建查询参数
        chunk_ids = [chunk['neo4j_id'] for chunk in chunks]
        
        # 使用高效的文本提取查询
        query = f"""
        UNWIND $chunk_ids AS id
        MATCH (c) WHERE id(c) = id
        RETURN id, c.{text_property} AS chunk_text
        """
        
        results = self.graph.query(query, params={"chunk_ids": chunk_ids})
        
        # 提取文本内容
        chunk_texts = []
        for row in results:
            text = row.get("chunk_text", "")
            # 确保文本不为空
            if not text:
                text = f"chunk_{row['id']}"
                
            chunk_texts.append(text)
        
        return chunk_texts
    
    def _update_embeddings_batch(self, chunks: List[Dict[str, Any]], 
                                embeddings: List[List[float]], 
                                embedding_property: str) -> None:
        """
        批量更新文本块embeddings
        
        Args:
            chunks: 文本块列表
            embeddings: 对应的embedding列表
            embedding_property: embedding属性名
        """
        # 构建更新数据
        update_data = []
        for i, chunk in enumerate(chunks):
            if i < len(embeddings) and embeddings[i] is not None:
                update_data.append({
                    "id": chunk['neo4j_id'],
                    "embedding": embeddings[i]
                })
        
        # 批量更新
        if update_data:
            try:
                query = f"""
                UNWIND $updates AS update
                MATCH (c) WHERE id(c) = update.id
                SET c.{embedding_property} = update.embedding
                """
                self.graph.query(query, params={"updates": update_data})
            except Exception as e:
                print(f"批量更新embeddings失败: {e}")
                # 回退到单个更新模式
                for update in update_data:
                    try:
                        single_query = f"""
                        MATCH (c) WHERE id(c) = $id
                        SET c.{embedding_property} = $embedding
                        """
                        self.graph.query(single_query, params={
                            "id": update["id"],
                            "embedding": update["embedding"]
                        })
                    except Exception as e2:
                        print(f"单个embedding更新失败 (ID: {update['id']}): {e2}")