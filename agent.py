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
from langchain_community.graphs import Neo4jGraph
from langchain_core.tools import tool
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

from model.get_models import get_llm_model, get_embeddings_model
from search.local_search import LocalSearch
from config.prompt import LC_SYSTEM_PROMPT, contextualize_q_system_prompt, MAP_SYSTEM_PROMPT, REDUCE_SYSTEM_PROMPT
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

level =0

# 全局检索器
@tool
def global_retriever(query: str) -> str:
    """回答有关网络小说《悟空传》的全局性问题。"""
    # 上面这段工具功能描述是必须的，否则调用工具会出错。
    
    # 检索MAP阶段生成中间结果的prompt与chain
    map_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                MAP_SYSTEM_PROMPT,
            ),
            (
                "human",
                """
                ---数据表格--- 
                {context_data}
                
                
                用户的问题是：
                {question}
                """,
            ),
        ]
    )
    map_chain = map_prompt | llm | StrOutputParser()

    # 连接Neo4j
    graph = Neo4jGraph(
        url=os.getenv('NEO4J_URI'),
        username=os.getenv('NEO4J_USERNAME'),
        password=os.getenv('NEO4J_PASSWORD'),
        refresh_schema=False,
    )
    # 检索指定层级的社区
    community_data = graph.query(
        """
        MATCH (c:__Community__)
        WHERE c.level = $level
        RETURN {communityId:c.id, full_content:c.full_content} AS output
        """,
        params={"level": level},
    )
    # 用LLM从每个检索到的社区摘要生成中间结果
    intermediate_results = []
    for community in tqdm(community_data, desc="Processing communities"):
        intermediate_response = map_chain.invoke(
            {"question": query, "context_data": community["output"]}
        )
        intermediate_results.append(intermediate_response)
        
    # 返回一个ToolMessage，包含每个社区对问题总结的要点，直接返回字典列表即可。
    return intermediate_results

# 工具列表有2个工具，局部检索器和全局检索器，LLM会根据工具的描述决定用哪一个。
tools = [lc_search_tool, global_retriever]

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
### Edges ----------------------------------------------------------------------

# 分流边
# 这个自定义的LangGraph边对工具调用的结果进行分流处理。
# 如果是全局检索，转到reduce结点生成回复。
# 如果是局部检索并且检索结果与问题相关，转到generate结点生成回复。
# 如果是局部检索并且检索结果与问题不相关，转到rewrite结点重构问题。
# 局部检索的结果是否与问题相关，提交给LLM去判断。
# 画流程图时要用到Literal。
def grade_documents(state) -> Literal["generate", "rewrite", "reduce"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    messages = state["messages"]
    # 倒数第2条消息是LLM发出工具调用请求的AIMessage。
    retrieve_message = messages[-2]
    
    # 如果是全局查询直接转去reduce结点。
    if retrieve_message.additional_kwargs["tool_calls"][0]["function"]["name"]== 'global_retriever':
        print("---Global retrieve---")
        return "reduce"

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
    # 最后一条消息是检索器返回的结果。
    last_message = messages[-1]
    # 在对话历史中，问题消息是倒数第3条。
    question = messages[-3].content
    
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})
    # LLM会给出检索结果与问题是否相关的判断, yes或no
    score = scored_result.content
    # 保险起见要转为小写！！！
    if score.lower() == "yes":
        print("---DECISION: DOCS RELEVANT---")
        print(score)
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

# 全局查询回复生成结点。
def reduce(state):
    """
    Generate answer for global retrieve

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---REDUCE---")
    messages = state["messages"]
    
    # 在对话历史中，问题消息是倒数第3条。
    question = messages[-3].content
    # 检索结果在最后一条消息中。
    last_message = messages[-1]
    docs = last_message.content

    # Reduce阶段生成最终结果的prompt与chain
    reduce_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                REDUCE_SYSTEM_PROMPT,
            ),
            (
                "human",
                """
                ---分析报告--- 
                {report_data}


                用户的问题是：
                {question}
                """,
            ),
        ]
    )
    reduce_chain = reduce_prompt | llm | StrOutputParser()

    # 再用LLM从每个社区摘要生成的中间结果生成最终的答复
    response = reduce_chain.invoke(
        {
            "report_data": docs,
            "question": question,
            "response_type": response_type,
        }
    )
    # 明确返回一条AIMessage。
    return {"messages": [AIMessage(content = response)]}

# Graph ------------------------------------------------------------------------
# 管理对话历史
memory = MemorySaver()

# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes we will cycle between
workflow.add_node("agent", agent)  # agent
retrieve = ToolNode(tools)
workflow.add_node("retrieve", retrieve)  # retrieval
workflow.add_node("rewrite", rewrite)  # Re-writing the question
workflow.add_node(
    "generate", generate
)  # Generating a response after we know the documents are relevant
# Call agent node to decide to retrieve or not
# 增加一个全局查询的reduce结点
workflow.add_node(
    "reduce", reduce
)

# 定义结点之间的连接
# 从agent结点开始
workflow.add_edge(START, "agent")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "agent",
    # Assess agent decision
    tools_condition,          # tools_condition()的输出是"tools"或END
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",  # 转到retrieve结点，执行局部检索或全局检索
        END: END,             # 直接结束
    },
)

# 检索结点执行结束后调边grade_documents，决定流转到哪个结点: generate、rewrite、reduce。
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
# 如果是局部查询生成，直接结束
workflow.add_edge("generate", END)
# 如果是重构问题，转到agent结点重新开始。
workflow.add_edge("rewrite", "agent")
# 增加一条全局查询到结束的边
workflow.add_edge("reduce", END)

# Compile
# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
graph = workflow.compile(checkpointer=memory)

# 画流程图
png_data = graph.get_graph().draw_mermaid_png()
# 指定将要保存的文件名
file_path = './assets/workflow.png'
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
ask_agent("悟空传》的主要人物有哪些？",graph,config)
print(get_answer(config))
ask_agent("他们最后的结局是什么？",graph,config)
print(get_answer(config))

# chat_history = memory.get(config)["channel_values"]["messages"]
# print(chat_history)