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

#from langchain_community.vectorstores import Chroma
# #from langchain_chroma import Chroma
# from langchain_community.vectorstores import Chroma
# from langchain_core.example_selectors import SemanticSimilarityExampleSelector
# from langchain_openai import OpenAIEmbeddings
# import streamlit as st

# @st.cache_resource
# def get_example_selector():

#     vector_store = Chroma(
#         collection_name="collections",  # Ensure a collection name is used
#         persist_directory="./chroma_store",  # Persist directory to keep Chroma data across sessions
#         embedding_function=OpenAIEmbeddings()
#     )
    
#     example_selector = SemanticSimilarityExampleSelector.from_examples(
#         examples,
#         OpenAIEmbeddings(),
#         vector_store,
#         k=2,
#         input_keys=["input"],
#     )

#     vector_store.persist()

#     return example_selector


import chromadb
from chromadb.utils import embedding_functions
import streamlit as st

# Use OpenAIEmbeddings from ChromaDB
openai_embeddings = embedding_functions.OpenAIEmbeddingFunction(api_key="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")

from langchain_core.example_selectors.base import BaseExampleSelector  # Make sure to import BaseExampleSelector
import chromadb

class CustomExampleSelector(BaseExampleSelector):
    def __init__(self, examples):
        self.examples = examples

    def add_example(self, example):
        """Add a new example to the list of examples."""
        self.examples.append(example)

    def select_examples(self, input_variables):
        """Select examples based on input variables."""
        new_word = input_variables["input"]
        new_word_length = len(new_word)

        best_match = None
        smallest_diff = float("inf")

        for example in self.examples:
            current_diff = abs(len(example["input"]) - new_word_length)

            if current_diff < smallest_diff:
                smallest_diff = current_diff
                best_match = example

        return [best_match]

@st.cache_resource
def get_example_selector():
    # Initialize ChromaDB client
    client = chromadb.Client()

    # Specify the persistence directory
    persist_directory = "./chroma_store"

    # Create or load a collection
    collection = client.get_or_create_collection(
        name="collections",
        embedding_function=openai_embeddings  # Ensure you have defined openai_embeddings
    )

    # Insert examples into the collection (if they are not already there)
    for example in examples:  # Make sure 'examples' is defined in your code
        input_text = example['input']
        query = example['query']
        
        # Check if this example already exists
        existing_docs = collection.get(ids=[input_text])
        if len(existing_docs["ids"]) == 0:  # If the document is not already in the collection
            collection.add(
                documents=[input_text],
                metadatas=[{"query": query}],
                ids=[input_text]
            )

    # Create a list of examples for the CustomExampleSelector
    stored_examples = [{'input': example['input'], 'query': example['query']} for example in examples]

    # Return an instance of CustomExampleSelector with the stored examples
    return CustomExampleSelector(stored_examples)