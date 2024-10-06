import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
import base64

import ast

import pyodbc

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
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from table_details import table_chain as select_table
from prompts import final_prompt, answer_prompt,input_prompt
import streamlit as st
from sqlalchemy import create_engine, text
import json

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

@st.cache_resource
def get_chain():
    print("Creating chain")
    tables_to_include = ['T24_ACCOUNTS', 'T24_CUSTOMERS_ALL']

    db = SQLDatabase.from_uri(
        f"mssql+pyodbc://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?driver=ODBC+Driver+17+for+SQL+Server",
        include_tables=tables_to_include
    )


    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    generate_query = create_sql_query_chain(llm, db, final_prompt) 
    execute_query = QuerySQLDataBaseTool(db=db)

    engine = create_engine(
        f"mssql+pyodbc://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?driver=ODBC+Driver+17+for+SQL+Server"
    )

    def execute_query_with_handling(query,history,max_retries=3):
        """
        Execute the query with error handling. If an error occurs, invoke the LLM to 
        generate a corrected query based on the error and retry until successful or max_retries is reached.
        
        Args:
        - query (str): The SQL query to execute.
        - history (ChatMessageHistory): The conversation history to provide context for the LLM.
        - max_retries (int): Maximum number of retries for query correction.
        
        Returns:
        - result (dict): Result of the query execution or error message after max retries.
        """
        query  = query.replace("`", "").strip()
        for attempt in range(max_retries):
            try:
                # Log the query and execute
                print(f"Attempt {attempt + 1}: Executing query: {query}")
                result= execute_query(query)
                with engine.connect() as connection:
                    result = connection.execute(text(query))
                    results_list = result.fetchall()
                    # Get column names
                    column_names = result.keys()  # This gives you a list of column names
                print("Column names sqlalchemy: ", column_names)
                # Ensure the result is a list
                if isinstance(result, dict) and 'result' in result:
                    results_list = result['result'] if isinstance(result['result'], list) else [result['result']]
                else:
                    results_list = results_list if isinstance(results_list, list) else [results_list]

                print(f"Attempt {attempt + 1}: Results_list: {results_list}")

                # If the query is successful, return the result
                if not (isinstance(results_list, str) and 'Error' in result):
                    return results_list

                # If error occurs, record the error message in history and log it
                error_message = f"SQL query error: {result}"
                print(f"Error on attempt {attempt + 1}: {error_message}")
                history.append({"role": "assistant", "content": error_message})

            except Exception as e:
                error_message = f"Unexpected error: {e}"
                print(f"Unexpected error on attempt {attempt + 1}: {error_message}")
                history.append({"role": "assistant", "content": error_message})

            # If the query failed, invoke the LLM to generate a corrected query
            if attempt < max_retries - 1:
                corrected_query = get_corrected_query_llm(query, history)
                query = corrected_query  # Update query with the new corrected query
                print(f"Attempt {attempt + 1}: Proposed Query:{query}")

            else:
                # Max retries reached, return the last error message
                return {'Summary': f"Query failed after {max_retries} attempts: {error_message}", 'Link': "", 'df': ""}

        return result
    
    def get_corrected_query_llm(query, history):
        """
        Use the LLM to analyze the query error and generate a corrected query.
        
        Args:
        - query (str): The original SQL query that failed.
        - history (ChatMessageHistory): The conversation history including error messages.
        
        Returns:
        - corrected_query (str): The corrected SQL query suggested by the LLM.
        """
        # Use the LLM to get a new query based on the error message and the query history
        prompt = f"""
        The following SQL query failed: {query}.
        Based on the history of queries and error messages below, suggest a corrected SQL query only:
        {history}

        Please return only the corrected SQL query without any additional information or sql or ```.
        """
      
        
        corrected_query_chain = (RunnableLambda(lambda _: prompt) | llm | StrOutputParser())
        corrected_query = corrected_query_chain.invoke({"question": prompt})
        
        print(f"LLM suggested query: {corrected_query}")
        
        return corrected_query

            

    chain = (
        RunnablePassthrough.assign(table_names_to_use=select_table) |
        RunnablePassthrough.assign(query=generate_query).assign(
            result=itemgetter("query", "messages") | RunnableLambda(lambda inputs: execute_query_with_handling(inputs[0], inputs[1]))
        ).assign(
            # Format the result and generate CSV download link
            Summary=lambda result: format_results_to_df(result)['Summary'],
            download_link = lambda result: format_results_to_df(result)['Link'],
            data = lambda result: format_results_to_df(result)['df']
        ) | custom_format
        
    )

    return chain



def custom_format(result):
    # Perform custom formatting here

    if isinstance(result, str):
        # If the result is already a string, just return it as the answer
        return result, None  # No download link in this case
    
    link = result['download_link']

    answer_chain = (answer_prompt | llm | StrOutputParser())
    answer = answer_chain.invoke(result)
    
    return answer ,link,str(result['data'])


def format_results_to_df(result):
    if 'Error' in result['result']:
        return {'Summary': f"Query failed: {result['result']}", 'Link': "", 'df': ""}

    if result['result']:
        try:
            result_tuples = result['result']
            print("parsing",result_tuples,len(result_tuples))
        except Exception as e:
            return {'Summary': f"Failed to parse result: {e}", 'Link': "", 'df': ""}
        
        if len(result_tuples) <= 5:
            return {'Summary': result_tuples, 'Link': "", 'df': ""}
        
        # More robust column extraction
        #columns = [col.strip() for col in result['query'].split('FROM')[0].replace('SELECT', '').split(',')]
        df = pd.DataFrame.from_records(result_tuples)  # Use extracted columns
        num_rows = len(df)
        num_cols = len(df.columns)

        download_link = generate_download_link(df) if not df.empty else ""
    
        summary = f"We have {num_rows} rows and {num_cols} columns in the result."
        print(summary)
        return {'Summary': summary, 'Link': download_link, 'df': result_tuples}
    else:
        return {'Summary': result['result'], 'Link': "", 'df': ""}





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
    history = create_history(messages)
    input_check = input_prompt | llm | StrOutputParser() | str
    answer = input_check.invoke({"question":question,"messages": history.messages})

    # Check if the question is related to data or it is a general question
    if answer != '1':
        return answer 
    
    chain = get_chain()
    response = chain.invoke({"question": question, "top_k": 3, "messages": history.messages})
    history.add_user_message(question)
    history.add_ai_message(response)
    return response




