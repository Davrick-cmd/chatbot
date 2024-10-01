
# Example data with fixed SQL syntax
examples = [
    {
        "input": "How many customers do we have in the bank?",
        "query": "SELECT COUNT(DISTINCT RECID) FROM V_CUSTOMER WHERE RECID IN (SELECT CUSTOMER FROM V_ACCOUNT WHERE CATEGORY <> '1080');"
    },
    {
        "input": "How many Current accounts?",
        "query": "SELECT COUNT(DISTINCT RECID) FROM V_ACCOUNT WHERE CATEGORY LIKE '1%';"
    },
    {
        "input": "How many accounts transacted yesterday?",
        "query": """SELECT COUNT(DISTINCT RECID) AS account_count, 
                    GREATEST(DATE_LAST_DR_BANK, DATE_LAST_CR_BANK, DATE_LAST_DR_CUST, DATE_LAST_CR_CUST) AS Last_transaction_date 
                    FROM V_ACCOUNT 
                    WHERE GREATEST(DATE_LAST_DR_BANK, DATE_LAST_CR_BANK, DATE_LAST_DR_CUST, DATE_LAST_CR_CUST) = CAST(GETDATE() - 1 AS DATE);"""
    },
    { 
        "input": "How many customers are in the retail segment?",
        "query": "SELECT COUNT(DISTINCT RECID) FROM V_CUSTOMER WHERE SEGMENT = '1';"
    },
    {
        "input": "How many Loan accounts?",
        "query": "SELECT COUNT(DISTINCT RECID) FROM V_ACCOUNT WHERE CATEGORY LIKE '3%';"
    },
    { 
        "input": "How many agriculture customers do we have?",
        "query": "SELECT COUNT(DISTINCT RECID) FROM V_CUSTOMER WHERE SEGMENT = '2';"
    },
    { 
        "input": "Give me a list of 5 customer names?",
        "query": "SELECT TOP 5 CUSTOMER_NAME FROM V_CUSTOMER;"
    },
    { 
        "input": "Give me a list of all vip customers?",
        "query": "SELECT * FROM V_CUSTOMER WHERE TARGET in ('57,'66','91');"
    },
    { 
        "input": "Give a list of all targets to identify different customers",
        "query": "SELECT DISTINCT TARGET FROM V_CUSTOMER;"
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