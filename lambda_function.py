import json
import os
import time
import datetime

import boto3
import asyncio

from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from langchain_community.vectorstores.falkordb_vector import dict_to_yaml_str
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAI
from langchain_aws import ChatBedrock
from typing import TypedDict, Annotated, List, Union

from langgraph.graph.message import add_messages
from langgraph.graph.message import AnyMessage
from langgraph.graph import StateGraph
from langchain_tavily import TavilySearch
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import START, END

from langgraph_checkpoint_dynamodb import DynamoDBTableConfig, DynamoDBConfig, DynamoDBSaver

RAG_PROMPT_TEMPLATE = """
당신은 사용자의 질문에 친절하고 상세하게 답변하는 AI 상담가입니다.
주어진 이전 대화 내용과 참고 정보를 바탕으로 사용자의 현재 질문에 답변해주세요.
사용자에게 정신상담과 관련된 질문까지 생성해주면 좋습니다.

[이전 대화 내용]
{messages}

[참고 정보]
{context}

[사용자 현재 질문]
{query}

[AI 답변]
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    query: str
    context: list
    # answer: str


# dynamodb = boto3.resource('dynamodb', endpoint_url='http://localhost:8000')
dynamodb = boto3.resource('dynamodb')

# TABLE_NAME = 'listen-you-full'
TABLE_NAME = os.environ['DYNAMODB_TABLE_NAME']
table = dynamodb.Table(TABLE_NAME)

# CHECKPOINT_TABLE_NAME = 'listen-you-checkpoints'
CHECKPOINT_TABLE_NAME = os.environ['DYNAMODB_CHECKPOINT_TABLE_NAME']

checkpoint_table_config = DynamoDBTableConfig(table_name=CHECKPOINT_TABLE_NAME)
checkpointer_config = DynamoDBConfig(table_config=checkpoint_table_config, endpoint_url='http://localhost:8000')
checkpointer = DynamoDBSaver(config=checkpointer_config, deploy=False)


def get_llm_local():
    llm = OpenAI(model="gpt-4o-mini")
    return llm

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

def generate(state: AgentState):
    query = state['query']
    messages = state['messages']
    context = state['context']

    chain = (rag_prompt
        | get_llm_local()
        | StrOutputParser())

    response = chain.invoke({'query': query, 'messages': messages, 'context': context, })

    return {'messages': [AIMessage(content=response)]}

def get_graph():
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node('generate', generate)

    graph_builder.add_edge(START, 'generate')
    graph_builder.add_edge('generate', END)
    return graph_builder.compile(checkpointer=checkpointer)

def insert_counsel_data(user_id, counsel_id, ai_answer, user_input):
    try:
        current_timestamp = str(datetime.datetime.now(datetime.timezone.utc).isoformat())

        response = table.put_item(
            Item={
                'user_id': user_id,
                'timestamp': current_timestamp,
                'counsel_id': counsel_id,
                'ai_answer': ai_answer,
                'user_input': user_input,
            }
        )
        print(f"Inserted {user_input} into {counsel_id}")
        return response
    except ClientError as e:
        print(f"데이터 삽입 중 오류 발생 : {e.response['Error']['Message']}")
        return None

def get_user_counsel_data(user_id):
    response = table.query(KeyConditionExpression=Key('user_id').eq(user_id))

    if response['Items']:
        for item in response['Items']:
            print(f"user_id: {item['user_id']}, timestamp: {item['timestamp']}, ai_answer: {item['ai_answer']}")


def get_counsel_data(counsel_id):
    response = table.query(IndexName='counsel_id-timestamp-index', KeyConditionExpression=Key('counsel_id').eq(counsel_id))

    if response['Items']:
        for item in response['Items']:
            print(f"user_id: {item['user_id']}, timestamp: {item['timestamp']}, ai_answer: {item['ai_answer']}")


def lambda_handler(event, context):
    load_dotenv()

    graph = get_graph()

    # state = {
    #     'query': event["query"],
    #     'context': ['당신은 전문 상담가 입니다. 사용자의 대답에 맞게 새로운 질문들을 만들어 내세요. 상대방이 위로를 받을 수 있어야 합니다.']
    # }

    query = event["query"]
    state = {
        'messages': [HumanMessage(query)],
        'query': query,
        'context': []
    }

    config = {"configurable": {"thread_id": event["counsel_id"]}}

    response = graph.invoke(state, config=config)

    insert_counsel_data(event["user_id"], event["counsel_id"], response["messages"][-1].content, event["query"])
    return {
        'message': response['messages'][-1].content
    }

if __name__ == "__main__":

    # 첫 번째 대화 (counsel_id="123")
    event1 = {'query': '배고파 밥 메뉴 추천좀','user_id': 1, 'counsel_id': "123"}
    print(f"\n--- 첫 번째 대화 시작 (counsel_id: {event1['counsel_id']}) ---")
    handler_response1 = lambda_handler(event1, None)
    print(f"AI 응답: {handler_response1['message']}")