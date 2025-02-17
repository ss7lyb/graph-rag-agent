from typing import List
from langsmith import traceable
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.tools.retriever import create_retriever_tool

from model.get_models import get_llm_model, get_embeddings_model
from search.local_search import LocalSearch
from config.prompt import LC_SYSTEM_PROMPT, contextualize_q_system_prompt
from config.settings import response_type, lc_description

class LocalSearchTool:
    def __init__(self):
        self.llm = get_llm_model()
        self.embeddings = get_embeddings_model()
        self.local_searcher = LocalSearch(self.llm, self.embeddings)
        self.retriever = self.local_searcher.as_retriever()
        self.chat_history = []
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

    @traceable
    def search(self, question: str, chat_history: List = None) -> str:
        """Execute local search with chat history context"""
        if chat_history is None:
            chat_history = self.chat_history
            
        ai_msg = self.rag_chain.invoke({
            "input": question,
            "response_type": response_type,
            "chat_history": chat_history,
        })
        return ai_msg["answer"]

    def get_tool(self):
        return create_retriever_tool(
            self.retriever,
            "lc_search_tool",
            lc_description,
        )