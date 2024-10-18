# Example data with fixed SQL syntax

import json

import chromadb
from chromadb.utils import embedding_functions
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.example_selectors.base import BaseExampleSelector  # Make sure to import BaseExampleSelector
import chromadb

# Function to read examples from a JSON file
def load_examples_from_json(file_path):
    with open(file_path, 'r') as file:
        examples = json.load(file)
    return examples


# Specify the path to your JSON file
json_file_path = 'examples_raw.json'

# Load the examples from the JSON file
examples = load_examples_from_json(json_file_path)

# Use OpenAIEmbeddings from ChromaDB
openai_embeddings = embedding_functions.OpenAIEmbeddingFunction(api_key="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")


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