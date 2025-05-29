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
    user_id: str
    query: str
    context: list
    answer: str
    survey_name: str


def get_llm():

    # 사용자가 제공한 모델 ID (정확한지 확인 필요)
    # 이 ID가 실제로 존재하는지, 그리고 사용자의 AWS 계정/리전에서 접근 가능한지 확인해야 합니다.
    model_id_from_user = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"

    # LangChain은 모델 ID에서 리전 정보를 직접 사용하지 않으므로,
    # 만약 'us.'가 리전을 의미한다면, Bedrock 클라이언트 생성 시 해당 리전을 지정해야 합니다.
    # 여기서는 우선 ID 자체의 유효성에 초점을 맞춥니다.
    # 가장 가능성이 높은 정확한 ID는 'anthropic.claude-3-sonnet-20240229-v1:0' 입니다.
    # 제공된 ID가 작동하지 않으면 공식 ID로 변경해 보세요.

    # ChatBedrock 인스턴스 생성
    # credentials_profile_name 와 region_name 은 환경에 맞게 설정
    llm = ChatBedrock(
        # credentials_profile_name="your-aws-profile", # AWS 프로파일 이름 (필요시)
        region_name="us-east-1",  # 모델이 있는 리전 (예: us-east-1)
        model_id=model_id_from_user,  # 또는 "anthropic.claude-3-sonnet-20240229-v1:0"
        model_kwargs={  # Claude 3에 맞는 추가 파라미터 (Messages API 형식)
            "anthropic_version": "bedrock-2023-05-31",  # Messages API 사용 명시
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
    graph_builder.add_node('decide_survey', decide_survey)
    graph_builder.add_node('generate', generate)

    graph_builder.add_edge(START, 'decide_survey')
    graph_builder.add_edge('decide_survey', 'generate')
    graph_builder.add_edge('generate', END)
    return graph_builder.compile()


def lambda_handler(event, context):
    graph = get_graph()

    state = {
        'query': event["query"],
        'survey_name': event["survey_name"]
    }

    response = graph.invoke(state)
    return {
        'message': response['answer']
    }