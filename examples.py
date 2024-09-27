examples = [
    {
        "input": "List all Retail customers with a current account.",
        "query": "SELECT c.RECCID, c.GIVEN_NAMES, c.FAMILY_NAME, a.CATEGORY, a.ACCOUNT_TITLE_D_1 FROM SDS_FBNK_CUSTOMER c JOIN SDS_FBNK_ACCOUNT a ON c.RECCID = a.CUSTOMER WHERE c.SEGMENT = '1' AND a.CATEGORY = '1001';"
    },
    {
        "input": "Get the highest online balance from any customer's account.",
        "query": "SELECT MAX(a.ONLINE_ACTUAL_BAL) AS MaxBalance FROM SDS_FBNK_ACCOUNT a;"
    },
    {
        "input": "Retrieve the details of customers born before 1980.",
        "query": "SELECT RECCID, GIVEN_NAMES, FAMILY_NAME, DATE_OF_BIRTH FROM SDS_FBNK_CUSTOMER WHERE DATE_OF_BIRTH < '1980-01-01';"
    },
    {
        "input": "List the accounts with a negative working balance.",
        "query": "SELECT CUSTOMER, ACCOUNT_TITLE_D_1, WORKING_BALANCE FROM SDS_FBNK_ACCOUNT WHERE WORKING_BALANCE < 0;"
    },
    {
        "input": "Find all customers whose legal ID is a 'NATIONAL.ID'.",
        "query": "SELECT RECCID, GIVEN_NAMES, FAMILY_NAME, LEGAL_ID FROM SDS_FBNK_CUSTOMER WHERE LEGAL_DOC_NAME = 'NATIONAL.ID';"
    },
    {
        "input": "Get the total number of accounts per currency type.",
        "query": "SELECT CURRENCY, COUNT(*) AS TotalAccounts FROM SDS_FBNK_ACCOUNT GROUP BY CURRENCY;"
    },
    {
        "input": "Find all customers who registered for internet banking services.",
        "query": "SELECT RECCID, GIVEN_NAMES, FAMILY_NAME FROM SDS_FBNK_CUSTOMER WHERE INTERNET_BANKING_SERVICE = 'YES';"
    },
    {
        "input": "List the details of the account with the largest debit amount.",
        "query": "SELECT CUSTOMER, ACCOUNT_TITLE_D_1, AMNT_LAST_DR_CUST FROM SDS_FBNK_ACCOUNT ORDER BY AMNT_LAST_DR_CUST DESC LIMIT 1;"
    }
]

from langchain_community.vectorstores import Chroma
#from langchain.vectorstores import Chroma
#from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
import streamlit as st

@st.cache_resource
def get_example_selector():
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        OpenAIEmbeddings(),
        Chroma,
        k=2,
        input_keys=["input"],
    )
    print(example_selector)
    return example_selector