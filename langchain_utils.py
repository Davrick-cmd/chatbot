import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport

import ast

import pyodbc

from typing import Union, Optional

import logging

# Set up logger
import os

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Set up logger with path to logs directory
conversation_logger = logging.getLogger('conversation')
conversation_logger.setLevel(logging.INFO)
handler = logging.FileHandler('logs/conversation.log')
handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
conversation_logger.addHandler(handler)


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
from datetime import datetime



# from pycaret.utils import get_config
from bokeh.plotting import figure
from bokeh.models import HoverTool



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

        return {'data':result,'Column_Names':column_names,'query':query}
    
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
            data_column=lambda result: result['formatted_result']['Column_names'],
            query=lambda result: result['formatted_result']['query']
        ) | custom_format
        
    )



    return chain


def custom_format(result):
    # Perform custom formatting here

    if isinstance(result, str):
        # If the result is already a string, just return it as the answer
        return result, None, None,None,None,None,None  # No download link in this case

    link = result['download_link']
    query = result['query']
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
        print(f"Results Query: {result['query']}\n")

        return response_str, link, str(result['data']),chart_type,str(column_names),str(data_column),str(query) # Add chart_type to the return values

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
        result_query = result['query']

        
        if len(result_tuples) <= 5:
            return {'Summary': result_tuples, 'Link': "", 'data_list': result_tuples,'Column_names':column_names,'query':result_query}

        # Convert result tuples into DataFrame
        df = pd.DataFrame.from_records(result_tuples,columns=column_names)
        download_link = generate_download_link(df) if not df.empty else ""
        # Generate a summary of rows and columns
        summary = f"We have {len(df)} rows and {len(df.columns)} columns."
        print(f"Summary {summary}\n")
        
        return {'Summary': summary, 'Link': download_link, 'data_list': result_tuples,'Column_names':column_names,'query':result_query}

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

def log_conversation_details(
    user_id: str, 
    question: str, 
    sql_query: str = None, 
    answer: str = None, 
    chart_type: str = None,
    feedback: str = None,
    feedback_comment: str = None
):
    """Log detailed conversation including SQL queries, visualization details, and user feedback if present"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "question": question,
        "sql_query": sql_query,
        "answer": answer,
        "chart_type": chart_type,
        "feedback": feedback,
        "feedback_comment": feedback_comment
    }
    try:
        conversation_logger.info(json.dumps(log_entry))
    except Exception as e:
        print(f"Logging error: {e}")

def invoke_chain(question, messages, user_id: str = "anonymous"):
    history = create_history(messages)
    input_check = input_prompt | llm_4 | StrOutputParser() | str
    answer = input_check.invoke({"question":question,"messages": history.messages})

    # Return early for non-data questions
    if answer != '1':
        return answer 
    
    # For data-related questions
    chain = get_chain()
    response = chain.invoke({"question": question, "top_k": 3, "messages": history.messages})
    
    history.add_user_message(question)
    history.add_ai_message(response)
    return response



def create_chart(chart_type: str, df: pd.DataFrame) -> None:
    """
    Create dynamic charts in Streamlit based on DataFrame analysis.
    
    Args:
        chart_type (str): Type of chart (bar, line, pie, scatter, histogram, box)
        df (pd.DataFrame): DataFrame to visualize
    """
    try:
        # Validate DataFrame
        if not isinstance(df, pd.DataFrame) or df.empty:
            st.error("Invalid or empty DataFrame")
            return

        # Infer data types
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        temporal_cols = df.select_dtypes(include=['datetime64']).columns

        if chart_type == "bar":
            if len(numeric_cols) < 1:
                st.error("Bar chart requires numeric columns")
                return
                
            x_col = st.selectbox("Select X-axis", df.columns)
            y_col = st.selectbox("Select Y-axis", numeric_cols)
            
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X(x_col, title=x_col),
                y=alt.Y(y_col, title=y_col, aggregate='sum'),
                tooltip=list(df.columns)
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

        elif chart_type == "line":
            if len(numeric_cols) < 1:
                st.error("Line chart requires numeric columns")
                return
                
            x_col = st.selectbox("Select X-axis", df.columns)
            y_col = st.selectbox("Select Y-axis", numeric_cols)
            
            if x_col in temporal_cols:
                chart = alt.Chart(df).mark_line().encode(
                    x=alt.X(x_col, title=x_col),
                    y=alt.Y(y_col, title=y_col),
                    tooltip=list(df.columns)
                ).interactive()
                st.altair_chart(chart, use_container_width=True)
            else:
                st.line_chart(df.set_index(x_col)[y_col])

        elif chart_type == "scatter":
            if len(numeric_cols) < 2:
                st.error("Scatter plot requires at least two numeric columns")
                return
                
            x_col = st.selectbox("Select X-axis", numeric_cols)
            y_col = st.selectbox("Select Y-axis", numeric_cols)
            
            chart = alt.Chart(df).mark_circle().encode(
                x=alt.X(x_col, title=x_col),
                y=alt.Y(y_col, title=y_col),
                tooltip=list(df.columns)
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

        elif chart_type == "pie":
            if len(categorical_cols) < 1 or len(numeric_cols) < 1:
                st.error("Pie chart requires one categorical and one numeric column")
                return
                
            cat_col = st.selectbox("Select category", categorical_cols)
            val_col = st.selectbox("Select values", numeric_cols)
            
            fig, ax = plt.subplots(figsize=(10, 10))
            df.groupby(cat_col)[val_col].sum().plot(kind='pie', ax=ax, autopct='%1.1f%%')
            st.pyplot(fig)
            plt.close()

        elif chart_type == "histogram":
            if len(numeric_cols) < 1:
                st.error("Histogram requires numeric columns")
                return
                
            num_col = st.selectbox("Select column", numeric_cols)
            bins = st.slider("Number of bins", 5, 100, 30)
            
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X(num_col, bin=alt.Bin(maxbins=bins)),
                y='count()'
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

        elif chart_type == "box":
            if len(numeric_cols) < 1:
                st.error("Box plot requires numeric columns")
                return
                
            num_col = st.selectbox("Select numeric column", numeric_cols)
            cat_col = None
            if len(categorical_cols) > 0:
                cat_col = st.selectbox("Select grouping column (optional)", 
                                     ["None"] + list(categorical_cols))
            
            chart = alt.Chart(df).mark_boxplot().encode(
                x=cat_col if cat_col and cat_col != "None" else alt.X('dummy:O', title=''),
                y=alt.Y(num_col, title=num_col)
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        print(f"Chart creation error: {type(e).__name__}: {str(e)}")





def create_interactive_visuals(data: Union[pd.DataFrame, list], target_column: Optional[str] = None) -> None:
    """
    Create interactive visuals using ydata-profiling and Bokeh.
    
    Args:
        data: DataFrame or list of data to visualize
        target_column: Optional target column for classification analysis
        
    Raises:
        ValueError: If data format is invalid or empty
    """
    try:
        # Data validation and conversion
        if isinstance(data, list):
            if not data:
                raise ValueError("Empty data list provided")
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            if data.empty:
                raise ValueError("Empty DataFrame provided")
            df = data.copy()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # Validate DataFrame columns
        if len(df.columns) < 2:
            raise ValueError("DataFrame must have at least two columns for visualization")

        # Generate EDA report with error handling
        try:
            with st.spinner("Generating EDA report..."):
                profile = ProfileReport(
                    df,
                    title="Exploratory Data Analysis",
                    explorative=True,
                    minimal=True,  # Faster processing for large datasets
                    correlations={
                        "auto": {"calculate": True},
                        "pearson": {"calculate": True},
                        "spearman": {"calculate": True}
                    },
                    plot={
                        "correlation": {
                            "cmap": "RdBu",
                            "bad": "#000000"
                        },
                        "missing": {
                            "cmap": "RdBu"
                        }
                    }
                )
                
                st.subheader("Exploratory Data Analysis Report")
                st_profile_report(profile)
        except Exception as e:
            st.error(f"Error generating EDA report: {str(e)}")
            print(f"EDA error: {type(e).__name__}: {str(e)}")

        # Create interactive Bokeh visualizations
        if target_column:
            try:
                # Validate target column
                if target_column not in df.columns:
                    raise ValueError(f"Target column '{target_column}' not found in DataFrame")

                unique_values = df[target_column].nunique()
                if unique_values <= 10:  # For categorical targets
                    # Create scatter plot with interactive features
                    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                    if len(numeric_cols) >= 2:
                        x_col = st.selectbox("Select X-axis", numeric_cols)
                        y_col = st.selectbox("Select Y-axis", numeric_cols)
                        
                        p = figure(
                            title=f"Scatter Plot: {x_col} vs {y_col}",
                            x_axis_label=x_col,
                            y_axis_label=y_col,
                            tools="pan,box_zoom,reset,save,hover",
                            sizing_mode="stretch_width",
                            height=400
                        )
                        
                        # Add interactive hover tool
                        hover = p.select_one(HoverTool)
                        hover.tooltips = [
                            ('X', f'@{x_col}'),
                            ('Y', f'@{y_col}'),
                            (target_column, f'@{target_column}')
                        ]
                        
                        # Create scatter plot with color coding
                        for value in df[target_column].unique():
                            mask = df[target_column] == value
                            p.scatter(
                                x=df[mask][x_col],
                                y=df[mask][y_col],
                                size=8,
                                alpha=0.6,
                                legend_label=str(value)
                            )
                        
                        p.legend.click_policy = "hide"
                        st.bokeh_chart(p)
                    else:
                        st.warning("Not enough numeric columns for scatter plot")
                else:
                    st.info(f"Target column has {unique_values} unique values. Scatter plot is only shown for <= 10 categories.")
                    
            except Exception as e:
                st.error(f"Error creating Bokeh visualization: {str(e)}")
                print(f"Bokeh error: {type(e).__name__}: {str(e)}")

    except Exception as e:
        st.error(f"Error in visualization process: {str(e)}")
        print(f"Visualization error: {type(e).__name__}: {str(e)}")










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