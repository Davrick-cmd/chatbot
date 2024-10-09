
# # Example data with fixed SQL syntax
# examples = [
#     {
#         "input": "How many customers do we have in the bank?",
#         "query": "SELECT COUNT(DISTINCT RECID) FROM V_CUSTOMER WHERE RECID IN (SELECT CUSTOMER FROM V_ACCOUNT WHERE CATEGORY <> '1080');"
#     },
#     {
#         "input": "How many Current accounts?",
#         "query": "SELECT COUNT(DISTINCT RECID) FROM V_ACCOUNT WHERE CATEGORY LIKE '1%';"
#     },
#     {
#         "input": "How many accounts transacted yesterday?",
#         "query": """SELECT COUNT(DISTINCT RECID) AS account_count, 
#                     GREATEST(DATE_LAST_DR_BANK, DATE_LAST_CR_BANK, DATE_LAST_DR_CUST, DATE_LAST_CR_CUST) AS Last_transaction_date 
#                     FROM V_ACCOUNT 
#                     WHERE GREATEST(DATE_LAST_DR_BANK, DATE_LAST_CR_BANK, DATE_LAST_DR_CUST, DATE_LAST_CR_CUST) = CAST(GETDATE() - 1 AS DATE);"""
#     },
#     { 
#         "input": "How many customers are in the retail segment?",
#         "query": "SELECT COUNT(DISTINCT RECID) FROM V_CUSTOMER WHERE SEGMENT = '1';"
#     },
#     {
#         "input": "How many Loan accounts?",
#         "query": "SELECT COUNT(DISTINCT RECID) FROM V_ACCOUNT WHERE CATEGORY LIKE '3%';"
#     },
#     { 
#         "input": "How many agriculture customers do we have?",
#         "query": "SELECT COUNT(DISTINCT RECID) FROM V_CUSTOMER WHERE SEGMENT = '2';"
#     },
#     { 
#         "input": "Give me a list of 10 corporate customers names?",
#         "query": "SELECT TOP 5 CUSTOMER_NAME FROM V_CUSTOMER WHERE SEGMENT = '3';"
#     },
#     { 
#         "input": "Give me a list of all vip customers?",
#         "query": "SELECT * FROM V_CUSTOMER WHERE TARGET in ('57,'66','91');"
#     },
#     { 
#         "input": "Give a list of all targets to identify different customers",
#         "query": "SELECT DISTINCT TARGET FROM V_CUSTOMER;"
#     }
# ]



# Example data with fixed SQL syntax
examples = [
    {
        "input": "How many customers do we have?",
        "query": """-- A customer is defined as someone who has an account with a category starting with '1' (e.g., Current) or '6' (e.g., Savings), 
                    -- and should not have an account in category '1080' (e.g., Suspended Accounts).
                    SELECT COUNT(DISTINCT CUSTOMER_NO) 
                    FROM T24_CUSTOMERS_ALL 
                    WHERE RECID IN (
                        SELECT CUSTOMER_NO 
                        FROM T24_ACCOUNTS 
                        WHERE CATEGORY <> '1080' 
                        AND SUBSTRING(CATEGORY, 1, 1) IN ('1', '6')
                    );"""
    },
    {
        "input": "How many accounts transacted yesterday?",
        "query": """SELECT COUNT(DISTINCT RECID) AS account_count, 
                    GREATEST(DATE_LAST_DR_BANK, DATE_LAST_CR_BANK, DATE_LAST_DR_CUST, DATE_LAST_CR_CUST) AS Last_transaction_date 
                    FROM T24_ACCOUNTS 
                    WHERE GREATEST(DATE_LAST_DR_BANK, DATE_LAST_CR_BANK, DATE_LAST_DR_CUST, DATE_LAST_CR_CUST) = CAST(GETDATE() - 1 AS DATE);"""
    },
    {
        "input":"How many active account do we have?",
        "query" : """SELECT COUNT(DISTINCT RECID)
                    FROM T24_ACCOUNTS
                    WHERE 
                        (
                            CATEGORY LIKE '1%' AND  -- Include current accounts
                            COALESCE(
                                CASE
                                    WHEN DATE_LAST_DR_CUST >= DATE_LAST_CR_CUST 
                                        AND DATE_LAST_DR_CUST >= DATE_LAST_CR_BANK 
                                        AND DATE_LAST_DR_CUST >= DATE_LAST_DR_BANK THEN DATE_LAST_DR_CUST
                                    WHEN DATE_LAST_CR_CUST >= DATE_LAST_DR_CUST 
                                        AND DATE_LAST_CR_CUST >= DATE_LAST_CR_BANK 
                                        AND DATE_LAST_CR_CUST >= DATE_LAST_DR_BANK THEN DATE_LAST_CR_CUST
                                    WHEN DATE_LAST_CR_BANK >= DATE_LAST_DR_CUST 
                                        AND DATE_LAST_CR_BANK >= DATE_LAST_CR_CUST 
                                        AND DATE_LAST_CR_BANK >= DATE_LAST_DR_BANK THEN DATE_LAST_CR_BANK
                                    ELSE DATE_LAST_DR_BANK
                                END, OPENING_DATE
                            ) >= DATEADD(DAY, -90, GETDATE())  --active for current accounts
                        )
                        OR
                        (
                            CATEGORY LIKE '6%' AND  -- Include savings accounts
                            COALESCE(
                                CASE
                                    WHEN DATE_LAST_DR_CUST >= DATE_LAST_CR_CUST 
                                        AND DATE_LAST_DR_CUST >= DATE_LAST_CR_BANK 
                                        AND DATE_LAST_DR_CUST >= DATE_LAST_DR_BANK THEN DATE_LAST_DR_CUST
                                    WHEN DATE_LAST_CR_CUST >= DATE_LAST_DR_CUST 
                                        AND DATE_LAST_CR_CUST >= DATE_LAST_CR_BANK 
                                        AND DATE_LAST_CR_CUST >= DATE_LAST_DR_BANK THEN DATE_LAST_CR_CUST
                                    WHEN DATE_LAST_CR_BANK >= DATE_LAST_DR_CUST 
                                        AND DATE_LAST_CR_BANK >= DATE_LAST_CR_CUST 
                                        AND DATE_LAST_CR_BANK >= DATE_LAST_DR_BANK THEN DATE_LAST_CR_BANK
                                    ELSE DATE_LAST_DR_BANK
                                END, OPENING_DATE
                            ) >= DATEADD(DAY, -720, GETDATE())  -- Active for savings accounts
                        );"""

    },
    {   "input":"How many Inactive accounts?",
        "query":"""SELECT COUNT(DISTINCT RECID)
                FROM T24_ACCOUNTS
                WHERE 
                    (CATEGORY LIKE '1%' AND  -- Include current accounts
                    COALESCE(
                        CASE
                            WHEN DATE_LAST_DR_CUST >= DATE_LAST_CR_CUST AND DATE_LAST_DR_CUST >= DATE_LAST_CR_BANK AND DATE_LAST_DR_CUST >= DATE_LAST_DR_BANK THEN DATE_LAST_DR_CUST
                            WHEN DATE_LAST_CR_CUST >= DATE_LAST_DR_CUST AND DATE_LAST_CR_CUST >= DATE_LAST_CR_BANK AND DATE_LAST_CR_CUST >= DATE_LAST_DR_BANK THEN DATE_LAST_CR_CUST
                            WHEN DATE_LAST_CR_BANK >= DATE_LAST_DR_CUST AND DATE_LAST_CR_BANK >= DATE_LAST_CR_CUST AND DATE_LAST_CR_BANK >= DATE_LAST_DR_BANK THEN DATE_LAST_CR_BANK
                            ELSE DATE_LAST_DR_BANK
                        END, OPENING_DATE) < DATEADD(DAY, -90, GETDATE())  -- Inactive for current accounts
                    AND 
                    COALESCE(
                        CASE
                            WHEN DATE_LAST_DR_CUST >= DATE_LAST_CR_CUST AND DATE_LAST_DR_CUST >= DATE_LAST_CR_BANK AND DATE_LAST_DR_CUST >= DATE_LAST_DR_BANK THEN DATE_LAST_DR_CUST
                            WHEN DATE_LAST_CR_CUST >= DATE_LAST_DR_CUST AND DATE_LAST_CR_CUST >= DATE_LAST_CR_BANK AND DATE_LAST_CR_CUST >= DATE_LAST_DR_BANK THEN DATE_LAST_CR_CUST
                            WHEN DATE_LAST_CR_BANK >= DATE_LAST_DR_CUST AND DATE_LAST_CR_BANK >= DATE_LAST_CR_CUST AND DATE_LAST_CR_BANK >= DATE_LAST_DR_BANK THEN DATE_LAST_CR_BANK
                            ELSE DATE_LAST_DR_BANK
                        END, OPENING_DATE) >= DATEADD(DAY, -180, GETDATE()))  -- At least one transaction in last 180 days
                    OR
                    (CATEGORY LIKE '6%' AND  -- Include savings accounts
                    COALESCE(
                        CASE
                            WHEN DATE_LAST_DR_CUST >= DATE_LAST_CR_CUST AND DATE_LAST_DR_CUST >= DATE_LAST_CR_BANK AND DATE_LAST_DR_CUST >= DATE_LAST_DR_BANK THEN DATE_LAST_DR_CUST
                            WHEN DATE_LAST_CR_CUST >= DATE_LAST_DR_CUST AND DATE_LAST_CR_CUST >= DATE_LAST_CR_BANK AND DATE_LAST_CR_CUST >= DATE_LAST_DR_BANK THEN DATE_LAST_CR_CUST
                            WHEN DATE_LAST_CR_BANK >= DATE_LAST_DR_CUST AND DATE_LAST_CR_BANK >= DATE_LAST_CR_CUST AND DATE_LAST_CR_BANK >= DATE_LAST_DR_BANK THEN DATE_LAST_CR_BANK
                            ELSE DATE_LAST_DR_BANK
                        END, OPENING_DATE) < DATEADD(DAY, -720, GETDATE()));  -- Savings accounts without transactions in last 720 days
                """

    },
    {
        "input": "How many customers are in the retail segment?",
        "query": """-- Segment '1' corresponds to Retail customers, 
                    -- Segment '2' corresponds to SME customers,
                    -- Segment '3' corresponds to Agriculture customers.
                    -- Segment '4' corresponds to Corporate customers.
                    SELECT COUNT(DISTINCT CUSTOMER_NO) 
                    FROM T24_CUSTOMERS_ALL 
                    WHERE SEGMENT = '1' 
                    AND RECID IN (
                        SELECT CUSTOMER_NO 
                        FROM T24_ACCOUNTS 
                        WHERE CATEGORY <> '1080' 
                        AND SUBSTRING(CATEGORY, 1, 1) IN ('1', '6')
                    );"""  # Fixed missing closing parenthesis
    },
    {
        "input": "How many customers are in the SME segment?",
        "query": """-- Segment '2' corresponds to SME (Small and Medium Enterprises) customers.
                    SELECT COUNT(DISTINCT CUSTOMER_NO) 
                    FROM T24_CUSTOMERS_ALL 
                    WHERE SEGMENT = '2'
                    AND RECID IN (
                        SELECT CUSTOMER_NO 
                        FROM T24_ACCOUNTS 
                        WHERE CATEGORY <> '1080' 
                        AND SUBSTRING(CATEGORY, 1, 1) IN ('1', '6')
                    );"""  # Fixed missing closing parenthesis
    },
    {
        "input": "How many customers are in the corporate segment?",
        "query": """-- Segment '4' corresponds to corporate customers.
                    SELECT COUNT(DISTINCT CUSTOMER_NO) 
                    FROM T24_CUSTOMERS_ALL 
                    WHERE SEGMENT = '4'
                    AND RECID IN (
                        SELECT CUSTOMER_NO 
                        FROM T24_ACCOUNTS 
                        WHERE CATEGORY <> '1080' 
                        AND SUBSTRING(CATEGORY, 1, 1) IN ('1', '6')
                    );"""  # Fixed missing closing parenthesis
    },
    {
        "input": "How many Loan accounts?",
        "query": """-- Categories starting with '3' correspond to loan accounts.
                    SELECT COUNT(DISTINCT RECID) 
                    FROM T24_ACCOUNTS 
                    WHERE CATEGORY LIKE '3%';"""
    },
    {
        "input": "How many Current accounts?",
        "query": """-- Categories starting with '1' correspond to current accounts.
                    SELECT COUNT(DISTINCT RECID) 
                    FROM T24_ACCOUNTS 
                    WHERE CATEGORY LIKE '1%';"""
    },
    {
        "input": "How many Savings accounts?",
        "query": """-- Categories starting with '6' correspond to savings accounts.
                    SELECT COUNT(DISTINCT RECID) 
                    FROM T24_ACCOUNTS 
                    WHERE CATEGORY LIKE '6%';"""
    },
    { 
        "input": "Give me a list of 10 corporate customers names?",
        "query": """-- Corporate customers are those in segment '4'.
                    SELECT TOP 10 SHORT_NAME 
                    FROM T24_CUSTOMERS_ALL 
                    WHERE SEGMENT = '4'
                    AND RECID IN (
                        SELECT CUSTOMER_NO 
                        FROM T24_ACCOUNTS 
                        WHERE CATEGORY <> '1080' 
                        AND SUBSTRING(CATEGORY, 1, 1) IN ('1', '6')
                    );"""  # Fixed SELECT TOP clause to 10
    },
    {
        "input": "Give me a list of all VIP customers?",
        "query": """-- VIP customers have TARGET values of '57', '66', or '91'.
                    SELECT * 
                    FROM T24_CUSTOMERS_ALL 
                    WHERE TARGET IN ('57', '66', '91');"""
    },
    {
        "input": "How many staff members do we have?",
        "query": """-- Staff members have a CATEGORY that starts with '1'.
                    SELECT COUNT(DISTINCT CUSTOMER_NO) 
                    FROM T24_CUSTOMERS_ALL 
                    WHERE TARGET = '1';"""
    },
    { 
        "input": "Give a list of all targets to identify different customers",
        "query": "SELECT DISTINCT TARGET FROM T24_CUSTOMERS_ALL;"
    }
]


import chromadb
from chromadb.utils import embedding_functions
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Use OpenAIEmbeddings from ChromaDB
openai_embeddings = embedding_functions.OpenAIEmbeddingFunction(api_key="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")

from langchain_core.example_selectors.base import BaseExampleSelector  # Make sure to import BaseExampleSelector
import chromadb

class CustomExampleSelector(BaseExampleSelector):
    def __init__(self, examples):
        self.examples = examples
        self.vectorizer = TfidfVectorizer()

        # Precompute the TF-IDF vectors for all the example inputs
        example_inputs = [example['input'] for example in self.examples]
        self.example_vectors = self.vectorizer.fit_transform(example_inputs)

    def add_example(self, example):
        """Add a new example and update the vectorized examples."""
        self.examples.append(example)
        new_input = example['input']
        new_vector = self.vectorizer.transform([new_input])
        self.example_vectors = self.vectorizer.fit_transform([*self.example_vectors.toarray(), new_vector.toarray()[0]])

    def select_examples(self, input_variables):
        """Select examples based on cosine similarity."""
        input_text = input_variables["input"]
        top_k = input_variables.get("top_k", 1)  # Get top_k, default to 1 if not provided

        # Vectorize the input text
        input_vector = self.vectorizer.transform([input_text])

        # Compute cosine similarities between the input and all example inputs
        similarities = cosine_similarity(input_vector, self.example_vectors).flatten()

        # Get the indices of the top_k most similar examples
        best_match_indices = similarities.argsort()[-top_k:][::-1]  # Sort and get top_k indices

        # Return the top_k matching examples
        return [self.examples[idx] for idx in best_match_indices]



# Example usage in get_example_selector:
@st.cache_resource
def get_example_selector():
    # Initialize ChromaDB client
    client = chromadb.Client()
    client.delete_collection

    # Specify the persistence directory
    persist_directory = "./chroma_store"

    # Create or load a collection
    collection = client.get_or_create_collection(
        name="collections",
        embedding_function=openai_embeddings  # Ensure you have defined openai_embeddings
    )


    # Insert examples into the collection (if they are not already there)
    for example in examples:
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