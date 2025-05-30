import json
import os
import boto3
import asyncio

from langchain_aws import ChatBedrock
from typing import TypedDict, Annotated, List, Union
from langgraph.graph.message import add_messages
from langgraph.graph.message import AnyMessage
from langgraph.graph import StateGraph
from langchain_tavily import TavilySearch
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import START, END

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    query: str
    context: list
    answer: str


def get_llm():
    model_id_from_user = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"

    llm = ChatBedrock(
        region_name="us-east-1",
        model_id=model_id_from_user,
        model_kwargs={
            "anthropic_version": "bedrock-2023-05-31",
            "temperature": 0.7,
            "max_tokens": 1000
        }
    )
    return llm

# def get_retriever():
#     embedding_function = UpstageEmbeddings(model="soloar-embedding-1-large",
#                                            api_key="up_oHwczQeYRxhIPy2wtDxc40qGSaA8p"
#                                            )

#     vector_store = Chroma(embedding_function=embedding_function,
#                           collection_name="mental_health_collection",
#                           persist_directory='./mental_health_collection'
#                           )

#     return vector_store.as_retriever(search_kwargs={"k": 3})

def generate(state: AgentState):
    generate_prompt = hub.pull("rlm/rag-prompt")

    context = state["context"]
    query = state["query"]
    rag_chain = generate_prompt | get_llm() | StrOutputParser()
    response = rag_chain.invoke({"context": context, "question": query})
    return {'answer': response}


def decide_survey(state: AgentState):
    tavily_tool = TavilySearch(max_results=5, search_depth="advanced", include_answer=True, include_raw_content=True, tavily_api_key= os.environ['TAVILY_API_KEY'])
    query = f"""
    아래 설문조사가 어떤 설문조사인지 밝히고,
    평가 기준을 검색해오세요.
    {state["survey_name"]}
    """
    response = tavily_tool.invoke({"query": query})
    return {'context': response}


def get_graph():
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node('generate', generate)

    graph_builder.add_edge(START, 'generate')
    graph_builder.add_edge('generate', END)
    return graph_builder.compile()


def lambda_handler(event, context):
    graph = get_graph()

    state = {
        'query': event["query"],
        'context': ['당신은 전문 상담가 입니다. 사용자의 대답에 맞게 새로운 질문들을 만들어 내세요. 상대방이 위로를 받을 수 있어야 합니다.']
    }

    response = graph.invoke(state)
    return {
        'message': response['answer']
    }

if __name__ == "__main__":
    event = {'query': 'what is the meaning of life?', 'survey_name': 'What is the meaning of life?',
             'survey_name': 'What is the meaning of life?'
             }
    lambda_handler(event, None)
