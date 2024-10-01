import os
from dotenv import load_dotenv
import pandas as pd
import base64

load_dotenv()

db_user = os.getenv("db_user")
db_password = os.getenv("db_password")
db_host = os.getenv("db_host")
db_name = os.getenv("db_name")
db_port =os.getenv('db_port')

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
#from langchain.memory import ChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import Runnable


from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from table_details import table_chain as select_table
from prompts import final_prompt, answer_prompt


import streamlit as st

@st.cache_resource
def get_chain():
    print("Creating chain")
    # Specify the tables you need
    tables_to_include = ['V_ACCOUNT', 'V_CUSTOMER']  # Replace these with your actual table names

    # SQL Server connection using pyodbc with specific tables included
    db = SQLDatabase.from_uri(
        f"mssql+pyodbc://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?driver=ODBC+Driver+17+for+SQL+Server",
        include_tables=tables_to_include, view_support=True  # Limit to specific tables
    )    

    # Define the language model and query generator
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    generate_query = create_sql_query_chain(llm, db, final_prompt) 
    execute_query = QuerySQLDataBaseTool(db=db)
    rephrase_answer = answer_prompt | llm | StrOutputParser()

    # Summary generation
    summarize_result = lambda df: {
        'num_rows': df.shape[0],
        'num_columns': df.shape[1]
    } if df is not None and not df.empty else {'num_rows': 0, 'num_columns': 0}


    # Replace ExtractQueryResults with RunnablePassthrough.assign
    chain = (
        RunnablePassthrough.assign(table_names_to_use=select_table) |
        RunnablePassthrough.assign(query=generate_query).assign(
            result=itemgetter("query") | execute_query
        ) | rephrase_answer
    )

    return chain

def create_history(messages):
    history = ChatMessageHistory()
    for message in messages:
        if message["role"] == "user":
            history.add_user_message(message["content"])
        else:
            history.add_ai_message(message["content"])
    return history



# def invoke_chain(question,messages):
#     chain = get_chain()
#     history = create_history(messages)
#     response = chain.invoke({"question": question,"top_k":3,"messages":history.messages})
#     history.add_user_message(question)
#     history.add_ai_message(response)
#     return response

# Function to create a chat message history from previous messages
def create_history(messages):
    history = ChatMessageHistory()
    for message in messages:
        if message["role"] == "user":
            history.add_user_message(message["content"])
        else:
            history.add_ai_message(message["content"])
    return history


# Function to convert query results to a DataFrame
def query_result_to_df(query_result):
    # Assuming query_result is a list of tuples or a dictionary
    if isinstance(query_result, list):
        df = pd.DataFrame(query_result)
    else:
        df = pd.DataFrame([query_result])
    return df


def generate_download_link(df=None):
    """Generate a download link for a CSV file of the DataFrame."""
    csv = df.to_csv(index=False)
    href = f'<a href="data:file/csv;base64,{base64.b64encode(csv.encode()).decode()}" download="query_result.csv">Download Data as CSV File</a>'
    return href

# Function to invoke the chain, get query results, and return a response
def invoke_chain(question, messages):
    # Get the chain (cached)
    chain = get_chain()

    # Create chat message history
    history = create_history(messages)

    # Invoke the chain with the question and message history
    response = chain.invoke({"question": question, "top_k": 3, "messages": history.messages})

    # Update the chat history with the new user message and AI response
    history.add_user_message(question)
    history.add_ai_message(response)

    return response