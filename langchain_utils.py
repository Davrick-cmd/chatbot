import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

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
from prompts import final_prompt, answer_prompt,input_prompt,check_query_prompt
import streamlit as st
from sqlalchemy import create_engine, text
import json

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
llm_4 = ChatOpenAI(model="gpt-4o", temperature=0.0)
llm_tune01 = ChatOpenAI(model="ft:gpt-4o-2024-08-06:personal:tune01:AJ4Ea2SL",temperature=0.0)
tables_to_include = ['ChatbotAccounts', 'BOT_CUSTOMER','BOT_FUNDS_TRANSFER']
db = SQLDatabase.from_uri(
        f"mssql+pyodbc://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?driver=ODBC+Driver+17+for+SQL+Server",
        include_tables=tables_to_include
    )

@st.cache_resource
def get_chain():
    print("Creating chain\n")
    tables_to_include = ['ChatbotAccounts', 'BOT_CUSTOMER','BOT_FUNDS_TRANSFER']

    db = SQLDatabase.from_uri(
        f"mssql+pyodbc://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?driver=ODBC+Driver+17+for+SQL+Server",
        include_tables=tables_to_include
    )


    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    generate_query = create_sql_query_chain(llm_tune01, db, final_prompt) #using a fine tuned model
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
        query = query.replace("`", "").replace("sql", "").strip()

        for attempt in range(max_retries):
            try:
                # Log the query and execute
                print(f"Attempt {attempt + 1}: Executing query: {query}\n")
                # result= execute_query(query)
                with engine.connect() as connection:
                    result = connection.execute(text(query))
                    results_list = result.fetchall()
                    # Get column names
                    column_names = result.keys()  # This gives you a list of column names
                # Ensure the result is a list
                if isinstance(result, dict) and 'result' in result:
                    results_list = result['result'] if isinstance(result['result'], list) else [result['result']]
                else:
                    results_list = results_list if isinstance(results_list, list) else [results_list]

                print(f"Attempt {attempt + 1}: Results: {results_list}\n")

                # If the query is successful, return the result
                if not (isinstance(results_list, str) and 'Error' in result):
                    return {'data':results_list,'Column_Names':column_names}

                # If error occurs, record the error message in history and log it
                error_message = f"SQL query error: {result}"
                print(f"Error on attempt {attempt + 1}: {error_message}\n")
                history.append({"role": "assistant", "content": error_message})

            except Exception as e:
                error_message = f"Unexpected error: {e}"
                print(f"Unexpected error on attempt {attempt + 1}: {error_message}\n")
                history.append({"role": "assistant", "content": error_message})

            # If the query failed, invoke the LLM to generate a corrected query
            if attempt < max_retries - 1:
                corrected_query = get_corrected_query_llm(query, history)
                query = corrected_query  # Update query with the new corrected query
                print(f"Attempt {attempt + 1}: LLM suggested query:{query}\n")

            else:
                # Max retries reached, return the last error message
                return {'Summary': f"Query failed after {max_retries} attempts: {error_message}", 'Link': "", 'df': ""}

        return {'data':result,'Column_Names':column_names}
    
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

        In your correction, please verify that all parentheses match, especially to avoid syntax errors.

        Please return only the corrected SQL query without any additional information or the word 'sql' or '```'.
        """

        corrected_query_chain = (RunnableLambda(lambda _: prompt) | llm | StrOutputParser())
        corrected_query = corrected_query_chain.invoke({"question": prompt})
        
        return corrected_query

            

    chain = (
        RunnablePassthrough.assign(table_names_to_use=select_table) |
        RunnablePassthrough.assign(query=generate_query).assign(
            result=itemgetter("question","query", "messages","table_names_to_use") | RunnableLambda(lambda inputs: check_against_definition(inputs[0],inputs[1],inputs[2],inputs[3])) |
            RunnableLambda(lambda inputs: execute_query_with_handling(inputs[0], inputs[1]))
        ).assign(formatted_result=lambda result: format_results_to_df(result))
        .assign(
            # Format the result and generate CSV download link
            Summary=lambda result: result['formatted_result']['Summary'],
            download_link=lambda result: result['formatted_result']['Link'],
            data=lambda result: result['formatted_result']['data_list'],
            data_column=lambda result: result['formatted_result']['Column_names']
        ) | custom_format
        
    )

    return chain


def custom_format(result):
    # Perform custom formatting here

    if isinstance(result, str):
        # If the result is already a string, just return it as the answer
        return result, None, None,None,None  # No download link in this case

    link = result['download_link']
    data_column = list(result['data_column'])

    answer_chain = (answer_prompt | llm)
    response = answer_chain.invoke(result)

    # Clean the response content to extract JSON
    cleaned_content = response.content.strip("```json\n").strip("\n```")  # Remove code block formatting

    # Attempt to load the cleaned response content as JSON
    try:
        response_content = json.loads(cleaned_content)  # Decode the cleaned string
        print(f"Response Content: {response_content}\n")

        response_str = response_content.get("Answer", "No answer provided.")  # Default message if key is missing
        chart_type = response_content.get("chart_type", "none")  # Default to "none" if key is missing
        column_names = response_content.get('Column_names',"")

        print(f"Chart Type: {chart_type}\n")

        return response_str, link, str(result['data']),chart_type,str(column_names),str(data_column) # Add chart_type to the return values

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}\n")
        return "Error processing response", link, str(result['data']),"none",""
    except Exception as e:
        print(f"An unexpected error occurred: {e}\n")
        return "Error processing response", link, str(result['data']),"none",""



def format_results_to_df(result):
    """
    Combines robust error handling and result parsing.
    
    Args:
    - result: The query result to process.

    Returns:
    - A dictionary with 'Summary', 'Link' to the download CSV, and 'df' as the DataFrame or parsed tuples.
    """
    if not result or 'Error' in result.get('result', {}).get('data', ''):
        return {'Summary': f"Query failed: {result['result']['data']}" if 'Error' in result.get('result', {}).get('data', '') else "No data returned", 
                'Link': "", 'data_list': ""}

    try:
        # Extract result tuples
        result_tuples = result['result']['data']
        # Generate a summary of rows and columns

        print(f"Parsing results of length {len(result_tuples)}\n")
        column_names = result['result']['Column_Names']
        
        if len(result_tuples) <= 5:
            return {'Summary': result_tuples, 'Link': "", 'data_list': result_tuples,'Column_names':column_names}

        # Convert result tuples into DataFrame
        df = pd.DataFrame.from_records(result_tuples,columns=column_names)
        download_link = generate_download_link(df) if not df.empty else ""
        # Generate a summary of rows and columns
        summary = f"We have {len(df)} rows and {len(df.columns)} columns."
        print(f"Summary {summary}\n")
        
        return {'Summary': summary, 'Link': download_link, 'data_list': result_tuples,'Column_names':column_names}

    except Exception as e:
        return {'Summary': f"Failed to parse result: {e}", 'Link': "", 'data_list': "",'Column_names':""}



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
    input_check = input_prompt | llm_4 | StrOutputParser() | str
    answer = input_check.invoke({"question":question,"messages": history.messages})

    # Check if the question is related to data or it is a general question
    print("answer :", answer)
    if answer != '1':
        return answer 
    
    chain = get_chain()
    response = chain.invoke({"question": question, "top_k": 3, "messages": history.messages})
    history.add_user_message(question)
    history.add_ai_message(response)
    return response




def create_chart(chart_type, data,column_names,data_columns):
    """
    Create a chart in Streamlit based on the chart type and data provided.
    
    Args:
        chart_type (str): The type of chart to create.
        data (list): The data to be used for the chart.
    """
    
    # Convert data to a DataFrame if it's in list format
    if isinstance(data, list):
        df = pd.DataFrame(data,columns=data_columns)
        df = df[data_columns]
        df.reset_index(drop=True, inplace=True) 
        print(f"Dataframe: {df}\n")
    else:
        st.error("Data format is not supported for chart creation.")
        return

    # Check the chart type and create the appropriate chart
    if chart_type == "bar":
        # Create a bar chart using Altair
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X(df.columns[0], title=column_names[0]),  # First column as x-axis
            y=alt.Y(df.columns[1], title=column_names[1])   # Second column as y-axis
        )
        st.altair_chart(chart, use_container_width=True)
    elif chart_type == "line":
        st.line_chart(df)  # Streamlit function for line chart
    elif chart_type == "pie":
        # Create a pie chart using Matplotlib
        plt.figure(figsize=(6, 6))
        plt.pie(df.iloc[:, 1], labels=df.iloc[:, 0], autopct='%1.1f%%')
        plt.title("Pie Chart")
        st.pyplot(plt)  # Streamlit function to display Matplotlib figures
    elif chart_type == "scatter":
        st.subheader("Scatter Plot")
        st.write(sns.scatterplot(data=df, x=df.columns[0], y=df.columns[1]))
        st.pyplot(plt)  # Display the figure
    elif chart_type == "histogram":
        st.subheader("Histogram")
        st.write(sns.histplot(df, bins=30))
        st.pyplot(plt)  # Display the figure
    elif chart_type == "box":
        st.subheader("Box Plot")
        st.write(sns.boxplot(data=df))
        st.pyplot(plt)  # Display the figure
    else:
        pass


def check_against_definition(question, query, chat_history,table_names):
    """
    Use the LLM to check the query against the definitions and adjust it if necessary.

    Args:
    - question (str): The natural language question that generated the query.
    - query (str): The generated SQL query to check.
    - chat_history (list): The chat history for context.
    - definitions (str): The predefined definitions for validation.
    - llm (OpenAI or similar model): The LLM to process and adjust the query if needed.

    Returns:
    - validated_query (str): The validated or adjusted query if it was out of line with definitions.
    """

    if len(table_names) == 0:
        table_names = tables_to_include
    

    temp_db =  SQLDatabase.from_uri(
        f"mssql+pyodbc://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?driver=ODBC+Driver+17+for+SQL+Server",
        include_tables=table_names)
    
    table_info = temp_db.get_table_info()

    # Use the LLM to check and potentially correct the query
    validated_query_chain = (check_query_prompt | llm_tune01 | StrOutputParser())
    
    # Invoke the chain and get the validated or corrected query
    validated_query = validated_query_chain.invoke({"question": question,"query":query,"messages":chat_history,"table_info":table_info})

    print(f"LLM Intial Query: {query}\n")

    print(f"LLM validated or corrected query against Definitions: {validated_query}\n")
    
    return query,chat_history
