import os
from dotenv import load_dotenv
import pandas as pd
import base64
from io import StringIO
import ast

load_dotenv()

db_user = os.getenv("db_user")
db_password = os.getenv("db_password")
db_host = os.getenv("db_host")
db_name = os.getenv("db_name")
db_port = os.getenv('db_port')

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.chat_message_histories import ChatMessageHistory
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from table_details import table_chain as select_table
from prompts import final_prompt, answer_prompt
import streamlit as st

@st.cache_resource
def get_chain():
    print("Creating chain")
    tables_to_include = ['V_ACCOUNT', 'V_CUSTOMER']

    db = SQLDatabase.from_uri(
        f"mssql+pyodbc://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?driver=ODBC+Driver+17+for+SQL+Server",
        include_tables=tables_to_include, view_support=True
    )    

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    generate_query = create_sql_query_chain(llm, db, final_prompt) 
    execute_query = QuerySQLDataBaseTool(db=db, verbose=False)
    rephrase_answer = answer_prompt | llm | StrOutputParser()

    chain = (
        RunnablePassthrough.assign(table_names_to_use=select_table) |
        RunnablePassthrough.assign(query=generate_query).assign(
            result=itemgetter("query") | execute_query
        ).assign(
            # Format the result and generate CSV download link
            formatted_output=lambda result: format_results_to_df(result)
        ) .assign(
            # Extract summary, download link, and dataframe without using |
            summary=lambda result: format_results_to_df(result)[0],  # First element is the summary
            download_link=lambda result: format_results_to_df(result)[1],  # Second element is the download link
            df=lambda result: format_results_to_df(result)[2]  # Third element is the DataFrame
        )
        | rephrase_answer
    )

    return chain


def format_results_to_df(result):
    """
    Format the SQL query result into a DataFrame and generate a download link for CSV.
    """

    print("Debug: Received result:", result) 
    # The result is in string format; convert it to a list of tuples
    if result['result'] != '':
        result_tuples = ast.literal_eval(result['result'])  # Safely evaluate the string to a Python object
        if len(result_tuples) <= 5:
            return result
        columns = result['query'].split()[3:]  # Extract column names from the query (assuming SELECT TOP N * FROM table)
        
        df = pd.DataFrame.from_records(result_tuples)
        print(df)
    else:
        return result

    num_rows = len(df)
    num_cols = len(df.columns)

    if not df.empty:
        download_link = generate_download_link(df)
    else:
        download_link = ""

    summary = f"We have {num_rows} rows and {num_cols} columns in the result."
    return summary, download_link, df


def generate_download_link(df):
    """
    Create a downloadable link for a CSV file from a DataFrame.
    """
    csv = df.to_csv(index=False)
    href = f'<a href="data:text/csv;base64,{base64.b64encode(csv.encode()).decode()}" download="query_result.csv">Download Data as CSV File</a>'
    return href

def create_history(messages):
    history = ChatMessageHistory()
    for message in messages:
        if message["role"] == "user":
            history.add_user_message(message["content"])
        else:
            history.add_ai_message(message["content"])
    return history

def invoke_chain(question, messages):
    chain = get_chain()
    history = create_history(messages)
    response = chain.invoke({"question": question, "top_k": 3, "messages": history.messages})
    history.add_user_message(question)
    history.add_ai_message(response)
    return response
