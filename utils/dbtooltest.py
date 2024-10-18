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
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool,InfoSQLDatabaseTool
from langchain_community.chat_message_histories import ChatMessageHistory
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda

import streamlit as st
from sqlalchemy import create_engine, text,MetaData,inspect
import json

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)


tables_to_include = ['T24_ACCOUNTS', 'T24_CUSTOMERS_ALL']

db = SQLDatabase.from_uri(
        f"mssql+pyodbc://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?driver=ODBC+Driver+17+for+SQL+Server",
        include_tables=tables_to_include
    )
engine = create_engine(
        f"mssql+pyodbc://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?driver=ODBC+Driver+17+for+SQL+Server"
    )

print(db.get_table_info())

def get_table_metadata(table_names):
    """
    Fetches table metadata including columns and sample rows for multiple tables.

    Args:
        table_names: A list of table names to get metadata for.

    Returns:
        A dictionary containing table names, columns, and sample_rows for each table.
    """
    # Create a metadata inspector
    inspector = inspect(engine)
    metadata = {}

    for table_name in table_names:
        # Get table columns
        columns = inspector.get_columns(table_name)
        column_names = [col['name'] for col in columns]

        # Optionally get sample rows (replace with your desired number)
        sample_rows = []
        with engine.connect() as conn:
            stmt = text(f"SELECT TOP 3 * FROM {table_name}")  # Use text() to make it executable
            result = conn.execute(stmt)
            
            # Create a list of dictionaries for the sample rows
            sample_rows = [{key: value for key, value in zip(result.keys(), row)} for row in result]

        # Store the metadata for the current table
        metadata[table_name] = {
            "columns": column_names,
            "sample_rows": sample_rows  # Optional
        }

    return metadata

# # Test the function with multiple tables
# table_info = get_table_metadata(['T24_ACCOUNTS', 'T24_CUSTOMERS'])
# for table, info in table_info.items():
#     print(f"Table: {table}")
#     print("Columns:", info['columns'])
#     print("Sample Rows:", info['sample_rows'])
#     print()