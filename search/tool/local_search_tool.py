from typing import List, Dict, Any
import hashlib
from langsmith import traceable
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.tools.retriever import create_retriever_tool
from langchain.utils.math import cosine_similarity

from model.get_models import get_llm_model, get_embeddings_model
from search.local_search import LocalSearch
from config.prompt import LC_SYSTEM_PROMPT, contextualize_q_system_prompt
from config.settings import response_type, lc_description

class LocalSearchCache:
    """本地搜索缓存"""
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
    
    def get_key(self, query: str, keywords: List[str] = None):
        """生成缓存键"""
        key_str = query
        if keywords:
            key_str += "||" + ",".join(sorted(keywords))
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, query: str, keywords: List[str] = None):
        """获取缓存结果"""
        key = self.get_key(query, keywords)
        return self.cache.get(key)
    
    def set(self, query: str, keywords: List[str], result: str):
        """设置缓存结果"""
        key = self.get_key(query, keywords)
        
        # 如果缓存已满，移除最早的项
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = result

class LocalSearchTool:
    def __init__(self):
        self.llm = get_llm_model()
        self.embeddings = get_embeddings_model()
        self.local_searcher = LocalSearch(self.llm, self.embeddings)
        self.retriever = self.local_searcher.as_retriever()
        self.chat_history = []
        self.cache = LocalSearchCache()
        self._setup_chains()

    def _setup_chains(self):
        # Context retriever prompt
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        # History aware retriever
        self.history_aware_retriever = create_history_aware_retriever(
            self.llm,
            self.retriever,
            contextualize_q_prompt,
        )

        # Local query prompt
        lc_prompt_with_history = ChatPromptTemplate.from_messages([
            ("system", LC_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", """
            ---分析报告--- 
            请注意，下面提供的分析报告按**重要性降序排列**。
            
            {context}
            
            用户的问题是：
            {input}

            请使用三级标题(###)标记主题
            """),
        ])

        # Question answer chain
        self.question_answer_chain = create_stuff_documents_chain(
            self.llm,
            lc_prompt_with_history,
        )

        # Full RAG chain
        self.rag_chain = create_retrieval_chain(
            self.history_aware_retriever,
            self.question_answer_chain,
        )

    def _filter_documents_by_relevance(self, docs, query: str) -> List:
        """根据相关性过滤文档"""
        # 使用向量相似度对文档进行排序
        try:
            vectorized_query = self.embeddings.embed_query(query)
            
            # 计算相似度和分数
            scored_docs = []
            for doc in docs:
                # 如果文档有向量表示
                if hasattr(doc, 'embedding') and doc.embedding:
                    similarity = cosine_similarity(vectorized_query, doc.embedding)
                else:
                    # 如果没有向量，给一个中等分数
                    similarity = 0.5
                    
                scored_docs.append({
                    'document': doc,
                    'score': similarity
                })
            
            # 按分数排序
            scored_docs.sort(key=lambda x: x['score'], reverse=True)
            
            # 只返回前5个高分文档
            top_docs = [item['document'] for item in scored_docs[:5]]
            return top_docs
        except Exception as e:
            print(f"文档过滤失败: {e}")
            return docs

    @traceable
    def search(self, question: str, keywords: List = None, chat_history: List = None) -> str:
        """Enhanced local search with keywords and chat history"""
        # 提取消息中的关键词，如果有的话
        if isinstance(question, dict) and "query" in question:
            if "keywords" in question:
                keywords = question["keywords"]
            question = question["query"]
            
        # Check cache first
        cached_result = self.cache.get(question, keywords)
        if cached_result:
            return cached_result
            
        if chat_history is None:
            chat_history = self.chat_history
            
        # 使用标准RAG链
        try:
            ai_msg = self.rag_chain.invoke({
                "input": question,
                "response_type": response_type,
                "chat_history": chat_history,
            })
            
            result = ai_msg.get("answer", "抱歉，我无法回答这个问题。")
            self.cache.set(question, keywords if keywords else [], result)
            return result
        except Exception as e:
            print(f"本地搜索失败: {e}")
            return f"搜索过程中出现问题: {str(e)}"

    def get_tool(self):
        return create_retriever_tool(
            self.retriever,
            "lc_search_tool",
            lc_description,
        )