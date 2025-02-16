import os

from typing import Annotated, Literal, Sequence, TypedDict

from langsmith import traceable
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.tools.retriever import create_retriever_tool
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.prebuilt import tools_condition
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
import pprint

from model.get_models import get_llm_model, get_embeddings_model
from search.local_search import LocalSearch
from config.prompt import LC_SYSTEM_PROMPT, contextualize_q_system_prompt
from config.settings import response_type

llm = get_llm_model()
embeddings = get_embeddings_model()

# 初始化LocalSearch实例--------------------------------------------------------
local_searcher = LocalSearch(llm, embeddings)
# 获取检索器
retriever = local_searcher.as_retriever()

# 建立可以处理对话历史上下文的 GraphRAG 查询Chain---------------------------

# 上下文检索器的提示词
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# 定义上下文检索器
history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt,
)

# 局部查询的提示词
lc_prompt_with_history = ChatPromptTemplate.from_messages([
    ("system", LC_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", """
    ---分析报告--- 
    请注意，下面提供的分析报告按**重要性降序排列**。
    
    {context}
    
    用户的问题是：
    {input}

    请按以下格式输出回答：
    1. 使用三级标题(###)标记主题
    2. 主要内容用清晰的段落展示
    3. 最后用"#### 引用数据"标记引用部分
    """),
])

# 定义局部查询的问答链
question_answer_chain_with_history = create_stuff_documents_chain(
    llm,
    lc_prompt_with_history,
)

# 定义支持上下文的局部查询问答链
rag_chain_with_history = create_retrieval_chain(
    history_aware_retriever,
    question_answer_chain_with_history,
)

# 对话记录
chat_history = []

# 定义可用LangSmith跟踪的局部查询函数
@traceable
def local_retriever(question, chat_history):
    ai_msg = rag_chain_with_history.invoke({
        "input": question,
        "response_type": "多个段落",
        "chat_history": chat_history,
    })
    return ai_msg["answer"]

# # 测试，有上下文
# question1 = "孙悟空跟如来佛祖之间有什么故事？"
# answer1 = local_retriever(question1, chat_history)
# print(answer1)
# chat_history.extend(
#     [
#         HumanMessage(content=question1),
#         AIMessage(content=answer1),
#     ]
# )
# # 这个问题需要知道上下文
# question2 = "最后的结局是什么？"
# answer2 = local_retriever(question2, chat_history)
# print(answer2)

# 定义agent-------------------------
# 我们需要自定义agent，否则无法修改系统提示词

lc_search_tool = create_retriever_tool(
    retriever,
    "lc_search_tool",
    "检索网络小说《悟空传》中各章节的人物与故事情节。",
)

# 工具列表中暂时只有一个工具局部检索器
tools = [lc_search_tool]

# 调试时如果要查看每个节点输入输出的状态，可以用这个函数插入打印的语句
def my_add_messages(left,right):
    print("\nLeft:\n")
    print(left)
    print("\nRight\n")
    print(right)
    return add_messages(left,right)
    
class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    # messages: Annotated[Sequence[BaseMessage], my_add_messages]
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Nodes and Edges --------------------------------------------------------------
### Edges

def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")

    # LLM
    model = llm

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )
    
    # Chain
    chain = prompt | llm

    messages = state["messages"]
    last_message = messages[-1]

    # 在对话历史中，问题消息是倒数第3条。
    question = messages[-3].content
    
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.content

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"

### Nodes

def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    model = llm

    model = model.bind_tools(tools)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    # 在对话历史中，问题消息是倒数第3条。
    question = messages[-3].content

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    # Rewriter
    model = llm

    response = model.invoke(msg)
    return {"messages": [response]}


def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    messages = state["messages"]
    # 在对话历史中，问题消息是倒数第3条。
    question = messages[-3].content
    
    last_message = messages[-1]

    docs = last_message.content

    # 局部查询的提示词
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                LC_SYSTEM_PROMPT,
            ),
            (
                "human",
                """
                ---分析报告--- 
                请注意，下面提供的分析报告按**重要性降序排列**。
                
                {context}
                

                用户的问题是：
                {question}
                """,
            ),
        ]
    )

    # Post-processing
    # def format_docs(docs):
    #     return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question, "response_type":response_type})
    
    # 这里有个Bug，response是String，generate节点返回的消息会自动判定为HumanMessage，其实是AIMessage。
    # 明确返回一条AIMessage。
    return {"messages": [AIMessage(content = response)]}

# Graph ------------------------------------------------------------------------
# 管理对话历史
memory = MemorySaver()

# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes we will cycle between
workflow.add_node("agent", agent)  # agent
retrieve = ToolNode([lc_search_tool])
workflow.add_node("retrieve", retrieve)  # retrieval
workflow.add_node("rewrite", rewrite)  # Re-writing the question
workflow.add_node(
    "generate", generate
)  # Generating a response after we know the documents are relevant
# Call agent node to decide to retrieve or not
workflow.add_edge(START, "agent")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "agent",
    # Assess agent decision
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

# Compile
# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
graph = workflow.compile(checkpointer=memory)

# 画流程图
png_data = graph.get_graph().draw_mermaid_png()
# 指定将要保存的文件名
file_path = './assets/lc_workflow.png'
# 打开一个文件用于写入二进制数据
with open(file_path, 'wb') as f:
    f.write(png_data)

# Run --------------------------------------------------------------------------
# 限制rewrite的次数，以免陷入无限的循环
config = {"configurable": {"thread_id": "226", "recursion_limit":5}}

def ask_agent(query,agent, config):
    inputs = {"messages": [("user", query)]}
    for output in agent.stream(inputs, config=config):
        for key, value in output.items():
            pprint.pprint(f"Output from node '{key}':")
            pprint.pprint("---")
            pprint.pprint(value, indent=2, width=80, depth=None)
        pprint.pprint("\n---\n")

def get_answer(config):
    chat_history = memory.get(config)["channel_values"]["messages"]
    answer = chat_history[-1].content
    return answer

ask_agent("你好，想问一些问题。",graph,config)
print(get_answer(config))
ask_agent("孙悟空和佛祖之间有什么故事？",graph,config)
print(get_answer(config))
ask_agent("他们最后的结局是什么？",graph,config)
print(get_answer(config))

chat_history = memory.get(config)["channel_values"]["messages"]
print(chat_history)