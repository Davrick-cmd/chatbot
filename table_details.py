import pandas as pd
import streamlit as st
from operator import itemgetter
from langchain.chains.openai_tools import create_extraction_chain_pydantic
#from langchain_core.pydantic_v1 import  Field
from pydantic import BaseModel,Field

from langchain_openai import ChatOpenAI



llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
from typing import List

# role = st.session_state.get("role", "User")  # Default to "User" if role is not set

# if role == "Admin":
#     tables_to_include = ['ChatbotAccounts', 'BOT_CUSTOMER', 'BOT_FUNDS_TRANSFER']
# else:
#     tables_to_include = ['ChatbotAccounts_og', 'BOT_CUSTOMER_2','BOT_FUNDS_TRANSFER']  

tables_to_include = ['ChatbotAccounts', 'BOT_CUSTOMER', 'BOT_FUNDS_TRANSFER']

@st.cache_data
def get_table_details():
    # Read the CSV file into a DataFrame
    table_description = pd.read_csv("database_table_descriptions.csv")

    # Filter the DataFrame to include only the specified tables
    table_description = table_description[table_description['Table'].isin(tables_to_include)]

    # print(table_description)
    table_docs = []

    # Iterate over the DataFrame rows to create Document objects
    table_details = ""
    for index, row in table_description.iterrows():
        table_details = table_details + "Table Name:" + row['Table'] + "\n" + "Table Description:" + row['Description'] + "\n\n"
    return table_details


class Table(BaseModel):
    """Table in SQL database."""

    name: str = Field(description="Name of table in SQL database.")

def get_tables(tables: List[Table]) -> List[str]:
    tables  = [table.name for table in tables]
    return tables


# table_names = "\n".join(db.get_usable_table_names())
table_details = get_table_details()

# Assuming context is passed from a chain, and table_details is fetched as shown earlier
table_details = get_table_details()

table_details_prompt = f"""
Return the names of ALL the SQL tables that MIGHT be relevant to the user question.

The tables are:
{table_details}

In addition to considering the tables, you must also take into account the following business definitions when identifying relevant tables:


{{context}}

Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed, and make sure that the tables you select align with these business definitions.
"""

table_chain = {
    "input": itemgetter("standalone_question"),
    "context": itemgetter('context'),  # Ensure 'context' is passed here
} | create_extraction_chain_pydantic(Table, llm, system_message=table_details_prompt) | get_tables
