# Example data with fixed SQL syntax
examples = [
    {
        "input": "How many customers do we have?",
        "query": """-- A customer is defined as someone who has an account with a category starting with '1' (e.g., Current) or '6' (e.g., Savings), 
                    -- and should not have an account in category '1080' (e.g., Suspended Accounts).
                SELECT COUNT(DISTINCT c.CUSTOMER_NO)
                FROM CB_CUSTOMERS c
                JOIN CB_ACCOUNTS a
                    ON c.CUSTOMER_NO = a.CUSTOMER_NUMBER
                WHERE a.CATEGORY NOT IN ('1080', '1031')
                AND SUBSTRING(a.CATEGORY, 1, 1) IN ('1', '6')

                    );"""
    },
    {
        "input": "How many accounts transacted yesterday?",
        "query": """SELECT COUNT(DISTINCT ACCOUNT_NUMBER) AS account_count 
                    FROM CB_ACCOUNTS 
                    WHERE LAST_TRANSACTION_DATE = CAST(GETDATE() - 1 AS DATE);"""
    },
    {
        "input":"How many active account do we have?",
        "query" : """SELECT COUNT(DISTINCT ACCOUNT_NUMBER)
                    FROM CB_ACCOUNTS
                    WHERE
                        (CATEGORY LIKE '1%' AND CATEGORY NOT IN ('1080', '1031') AND LAST_TRANSACTION_DATE >= DATEADD(DAY, -90, GETDATE())) -- Active for current accounts
                        OR
                        (CATEGORY LIKE '6%' AND LAST_TRANSACTION_DATE >= DATEADD(DAY, -360, GETDATE())) -- Active for savings accounts
                        AND CATEGORY NOT IN ('1080', '1031') --Unsecured Financing Accounts
                    """

    },
    {   "input":"How many Inactive accounts?",
        "query":"""SELECT COUNT(DISTINCT ACCOUNT_NUMBER)
                FROM CB_ACCOUNTS
                WHERE 
                    (CATEGORY LIKE '1%' AND  CATEGORY NOT IN ('1080', '1031') AND LAST_TRANSACTION_DATE < DATEADD(DAY, -90, GETDATE())  -- Inactive for current accounts
                    AND LAST_TRANSACTION_DATE >= DATEADD(DAY, -180, GETDATE()))  -- At least one transaction in last 180 days
                    OR
                    (CATEGORY LIKE '6%' AND  LAST_TRANSACTION_DATE < DATEADD(DAY, -360, GETDATE()));  -- Savings accounts without transactions in last 360 days
                """

    },
    {
        "input": "How many customers are in the retail segment?",
        "query": """-- Segment '1' corresponds to Retail customers, 
                    -- Segment '2' corresponds to SME customers,
                    -- Segment '3' corresponds to Agriculture customers.
                    -- Segment '4' corresponds to Corporate customers.
                        SELECT COUNT(DISTINCT c.CUSTOMER_NO)
                        FROM CB_CUSTOMERS c
                        JOIN CB_ACCOUNTS a
                            ON c.CUSTOMER_NO = a.CUSTOMER_NUMBER
                        WHERE c.SEGMENT = '1'
                        AND a.CATEGORY NOT IN ('1080', '1031')
                        AND SUBSTRING(a.CATEGORY, 1, 1) IN ('1', '6');
                    """  # Fixed missing closing parenthesis
    },
    {
        "input": "How many customers are in the SME segment?",
        "query": """-- Segment '2' corresponds to SME (Small and Medium Enterprises) customers.
                        SELECT COUNT(DISTINCT c.CUSTOMER_NO)
                        FROM CB_CUSTOMERS c
                        JOIN CB_ACCOUNTS a
                            ON c.CUSTOMER_NO = a.CUSTOMER_NUMBER
                        WHERE c.SEGMENT = '2'
                        AND a.CATEGORY NOT IN ('1080', '1031')
                        AND SUBSTRING(a.CATEGORY, 1, 1) IN ('1', '6');"""  # Fixed missing closing parenthesis
    },
    {
        "input": "How many customers are in the corporate segment?",
        "query": """-- Segment '4' corresponds to corporate customers.
                        SELECT COUNT(DISTINCT c.CUSTOMER_NO)
                        FROM CB_CUSTOMERS c
                        JOIN CB_ACCOUNTS a
                            ON c.CUSTOMER_NO = a.CUSTOMER_NUMBER
                        WHERE c.SEGMENT = '4'
                        AND a.CATEGORY NOT IN ('1080', '1031')
                        AND SUBSTRING(a.CATEGORY, 1, 1) IN ('1', '6');"""  # Fixed missing closing parenthesis
    },
    {
        "input": "How many Loan accounts?",
        "query": """-- Categories starting with '3' correspond to loan accounts.
                    SELECT COUNT(DISTINCT ACCOUNT_NUMBER) 
                    FROM CB_ACCOUNTS 
                    WHERE CATEGORY LIKE '3%';"""
    },
    {
        "input": "How many Current accounts?",
        "query": """-- Categories starting with '1' correspond to current accounts.
                    SELECT COUNT(DISTINCT ACCOUNT_NUMBER) 
                    FROM CB_ACCOUNTS 
                    WHERE (CATEGORY LIKE '1%' AND CATEGORY NOT IN ('1080', '1031'));"""
    },
    {
        "input": "How many Savings accounts?",
        "query": """-- Categories starting with '6' correspond to savings accounts.
                    SELECT COUNT(DISTINCT ACCOUNT_NUMBER) 
                    FROM CB_ACCOUNTS 
                    WHERE CATEGORY LIKE '6%'
                ;"""
    },
    { 
        "input": "Give me a list of 10 corporate customers names?",
        "query": """-- Corporate customers are those in segment '4'.
                    SELECT TOP 10 c.SHORT_NAME
                    FROM CB_CUSTOMERS c
                    JOIN CB_ACCOUNTS a
                        ON c.CUSTOMER_NO = a.CUSTOMER_NUMBER
                    WHERE c.SEGMENT = '4'
                    AND a.CATEGORY <> '1080'
                    AND SUBSTRING(a.CATEGORY, 1, 1) IN ('1', '6')
                ;"""  # Fixed SELECT TOP clause to 10
    },
    {
        "input": "Give me a list of all VIP customers?",
        "query": """-- VIP customers have TARGET values of '57', '66', or '91'.
                    SELECT * 
                    FROM CB_CUSTOMERS 
                    WHERE TARGET IN ('57', '66', '91');"""
    },
    {
        "input": "How many staff members do we have?",
        "query": """-- Staff members have a CATEGORY that starts with '1'.
                    SELECT COUNT(DISTINCT CUSTOMER_NO) 
                    FROM CB_CUSTOMERS 
                    WHERE TARGET = '1';"""
    },
    { 
        "input": "Give a list of all targets to identify different customers",
        "query": "SELECT DISTINCT TARGET FROM CB_CUSTOMERS;"
    },
    {
        "input":"How many inactive customers do we have?",
        "query":""" --Inactive Customer": "A customer with no active accounts but possessing at least one inactive current or savings account.
                    WITH Inactive_Accounts AS (
                        SELECT CUSTOMER_NUMBER
                        FROM CB_ACCOUNTS
                        WHERE 
                            -- Inactive current account (last transaction between 90 and 180 days ago)
                            (CATEGORY LIKE '1%' AND CATEGORY NOT IN ('1080', '1031')
                            AND LAST_TRANSACTION_DATE < DATEADD(DAY, -90, GETDATE())
                            AND LAST_TRANSACTION_DATE >= DATEADD(DAY, -180, GETDATE()))
                            
                            OR
                            
                            -- Inactive savings account (no transactions for more than 360 days)
                            (CATEGORY LIKE '6%' AND CATEGORY NOT IN ('1080', '1031')
                            AND LAST_TRANSACTION_DATE < DATEADD(DAY, -360, GETDATE()))
                    ),
                    Active_Accounts AS (
                        SELECT CUSTOMER_NUMBER
                        FROM CB_ACCOUNTS
                        WHERE 
                            -- Active current accounts
                            (CATEGORY LIKE '1%' AND CATEGORY NOT IN ('1080', '1031')
                            AND LAST_TRANSACTION_DATE >= DATEADD(DAY, -90, GETDATE()))
                            
                            OR
                            
                            -- Active savings accounts
                            (CATEGORY LIKE '6%' AND CATEGORY NOT IN ('1080', '1031')
                            AND LAST_TRANSACTION_DATE >= DATEADD(DAY, -360, GETDATE()))
                    )
                    SELECT COUNT(DISTINCT C.CUSTOMER_NO) AS Inactive_Customers
                    FROM CB_CUSTOMERS C
                    -- Join to ensure customer has at least one inactive account
                    INNER JOIN Inactive_Accounts IA
                        ON C.CUSTOMER_NO = IA.CUSTOMER_NUMBER
                    -- Left join to filter out customers with active accounts
                    LEFT JOIN Active_Accounts AA
                        ON C.CUSTOMER_NO = AA.CUSTOMER_NUMBER
                    WHERE AA.CUSTOMER_NUMBER IS NULL; -- Ensure no active accounts;
                    """
    },
    {   "input":"How many dom closed customers do we have?",
        "query":""" --Dom Closed Customer": "A customer with no active, inactive, or dormant accounts but possessing at least one dom closed current account.
                    WITH Dom_Closed_Accounts AS (
                        SELECT CUSTOMER_NUMBER
                        FROM CB_ACCOUNTS
                        WHERE 
                            -- Dom Closed Account (Inactive for over 360 days but had transactions in the last 1800 days)
                            CATEGORY LIKE '1%' 
                            AND CATEGORY NOT IN ('1080', '1031')
                            AND LAST_TRANSACTION_DATE < DATEADD(DAY, -360, GETDATE()) 
                            AND LAST_TRANSACTION_DATE >= DATEADD(DAY, -1800, GETDATE())
                    ),
                    Active_Accounts AS (
                        SELECT CUSTOMER_NUMBER
                        FROM CB_ACCOUNTS
                        WHERE 
                            -- Active current account
                            (CATEGORY LIKE '1%' AND CATEGORY NOT IN ('1080', '1031')
                            AND LAST_TRANSACTION_DATE >= DATEADD(DAY, -90, GETDATE()))
                            
                            OR
                            
                            -- Active savings account
                            (CATEGORY LIKE '6%' AND LAST_TRANSACTION_DATE >= DATEADD(DAY, -360, GETDATE()))
                    ),
                    Inactive_Accounts AS (
                        SELECT CUSTOMER_NUMBER
                        FROM CB_ACCOUNTS
                        WHERE 
                            -- Inactive current account
                            (CATEGORY LIKE '1%' AND CATEGORY NOT IN ('1080', '1031')
                            AND LAST_TRANSACTION_DATE < DATEADD(DAY, -90, GETDATE())
                            AND LAST_TRANSACTION_DATE >= DATEADD(DAY, -180, GETDATE()))
                            
                            OR
                            
                            -- Inactive savings account
                            (CATEGORY LIKE '6%' AND LAST_TRANSACTION_DATE < DATEADD(DAY, -360, GETDATE())
                            AND LAST_TRANSACTION_DATE >= DATEADD(DAY, -720, GETDATE()))
                    ),
                    Dormant_Accounts AS (
                        SELECT CUSTOMER_NUMBER
                        FROM CB_ACCOUNTS
                        WHERE 
                            -- Dormant current account
                            CATEGORY LIKE '1%' 
                            AND CATEGORY NOT IN ('1080', '1031')
                            AND LAST_TRANSACTION_DATE < DATEADD(DAY, -180, GETDATE())
                            AND LAST_TRANSACTION_DATE >= DATEADD(DAY, -360, GETDATE())
                    )
                    SELECT COUNT(DISTINCT C.CUSTOMER_NO) AS Dom_Closed_Customers
                    FROM CB_CUSTOMERS C
                    -- Join with Dom Closed Accounts to ensure customer has at least one dom closed account
                    INNER JOIN Dom_Closed_Accounts DCA
                        ON C.CUSTOMER_NO = DCA.CUSTOMER_NUMBER
                    -- Left join with Active Accounts to filter out customers with active accounts
                    LEFT JOIN Active_Accounts AA
                        ON C.CUSTOMER_NO = AA.CUSTOMER_NUMBER
                    -- Left join with Inactive Accounts to filter out customers with inactive accounts
                    LEFT JOIN Inactive_Accounts IA
                        ON C.CUSTOMER_NO = IA.CUSTOMER_NUMBER
                    -- Left join with Dormant Accounts to filter out customers with dormant accounts
                    LEFT JOIN Dormant_Accounts DA
                        ON C.CUSTOMER_NO = DA.CUSTOMER_NUMBER
                    WHERE AA.CUSTOMER_NUMBER IS NULL
                    AND IA.CUSTOMER_NUMBER IS NULL
                    AND DA.CUSTOMER_NUMBER IS NULL; -- Ensure no active, inactive, or dormant accounts;

        """

    },
    {
        "input":"What is our churn rate number?",
        "query":"""
                WITH Dom_Closed_Accounts AS (
                    SELECT CUSTOMER_NUMBER
                    FROM CB_ACCOUNTS
                    WHERE 
                        -- Dom Closed Account (Inactive for over 360 days but had transactions in the last 1800 days)
                        CATEGORY LIKE '1%' 
                        AND CATEGORY NOT IN ('1080', '1031')
                        AND LAST_TRANSACTION_DATE < DATEADD(DAY, -360, GETDATE()) 
                        AND LAST_TRANSACTION_DATE >= DATEADD(DAY, -1800, GETDATE())
                ),
                Active_Accounts AS (
                    SELECT CUSTOMER_NUMBER
                    FROM CB_ACCOUNTS
                    WHERE 
                        -- Active current account
                        (CATEGORY LIKE '1%' AND CATEGORY NOT IN ('1080', '1031')
                        AND LAST_TRANSACTION_DATE >= DATEADD(DAY, -90, GETDATE()))
                        
                        OR
                        
                        -- Active savings account
                        (CATEGORY LIKE '6%' AND LAST_TRANSACTION_DATE >= DATEADD(DAY, -360, GETDATE()))
                ),
                Inactive_Accounts AS (
                    SELECT CUSTOMER_NUMBER
                    FROM CB_ACCOUNTS
                    WHERE 
                        -- Inactive current account
                        (CATEGORY LIKE '1%' AND CATEGORY NOT IN ('1080', '1031')
                        AND LAST_TRANSACTION_DATE < DATEADD(DAY, -90, GETDATE())
                        AND LAST_TRANSACTION_DATE >= DATEADD(DAY, -180, GETDATE()))
                        
                        OR
                        
                        -- Inactive savings account
                        (CATEGORY LIKE '6%' AND LAST_TRANSACTION_DATE < DATEADD(DAY, -360, GETDATE())
                        AND LAST_TRANSACTION_DATE >= DATEADD(DAY, -720, GETDATE()))
                ),
                Dormant_Accounts AS (
                    SELECT CUSTOMER_NUMBER
                    FROM CB_ACCOUNTS
                    WHERE 
                        -- Dormant current account
                        CATEGORY LIKE '1%' 
                        AND CATEGORY NOT IN ('1080', '1031')
                        AND LAST_TRANSACTION_DATE < DATEADD(DAY, -180, GETDATE())
                        AND LAST_TRANSACTION_DATE >= DATEADD(DAY, -360, GETDATE())
                ),
                Total_Customers AS (
                    SELECT COUNT(DISTINCT c.CUSTOMER_NO) AS Total
                    FROM CB_CUSTOMERS c
                    JOIN CB_ACCOUNTS a
                        ON c.CUSTOMER_NO = a.CUSTOMER_NUMBER
                    WHERE a.CATEGORY NOT IN ('1080', '1031')
                    AND SUBSTRING(a.CATEGORY, 1, 1) IN ('1', '6')
                )

                SELECT 
                    COUNT(DISTINCT C.CUSTOMER_NO) AS Dom_Closed_Customers,
                    (SELECT Total FROM Total_Customers) AS Total_Customers,
                    CASE 
                        WHEN (SELECT Total FROM Total_Customers) > 0 THEN 
                            (COUNT(DISTINCT C.CUSTOMER_NO) * 1.0 / (SELECT Total FROM Total_Customers)) * 100
                        ELSE 0
                    END AS Churn_Rate
                FROM CB_CUSTOMERS C
                -- Join with Dom Closed Accounts to ensure customer has at least one dom closed account
                INNER JOIN Dom_Closed_Accounts DCA
                    ON C.CUSTOMER_NO = DCA.CUSTOMER_NUMBER
                -- Left join with Active Accounts to filter out customers with active accounts
                LEFT JOIN Active_Accounts AA
                    ON C.CUSTOMER_NO = AA.CUSTOMER_NUMBER
                -- Left join with Inactive Accounts to filter out customers with inactive accounts
                LEFT JOIN Inactive_Accounts IA
                    ON C.CUSTOMER_NO = IA.CUSTOMER_NUMBER
                -- Left join with Dormant Accounts to filter out customers with dormant accounts
                LEFT JOIN Dormant_Accounts DA
                    ON C.CUSTOMER_NO = DA.CUSTOMER_NUMBER
                -- Ensure no active, inactive, or dormant accounts
                WHERE AA.CUSTOMER_NUMBER IS NULL
                AND IA.CUSTOMER_NUMBER IS NULL
                AND DA.CUSTOMER_NUMBER IS NULL;

        """
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