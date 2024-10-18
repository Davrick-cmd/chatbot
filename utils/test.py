import os
from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings  # Updated import from langchain_openai

# Your example queries
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
        "input": "Find all customers whose legal ID is a 'NATIONAL.ID'.",
        "query": "SELECT RECCID, GIVEN_NAMES, FAMILY_NAME, LEGAL_ID FROM SDS_FBNK_CUSTOMER WHERE LEGAL_DOC_NAME = 'NATIONAL.ID';"
    },
]

# Set your OpenAI API key (either via environment variable or directly as a parameter)
openai_api_key = os.getenv('OPENAI_API_KEY', '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')

def get_example_selector():
    # Initialize the embeddings with the API key
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Create example selector based on semantic similarity
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        embeddings,  # Pass embeddings
        Chroma,  # Specify the vectorstore class (Chroma)
        k=2,
        input_keys=["input"]
    )

    return example_selector

# Function to return a similar query based on input
def get_similar_query(input_text):
    # Get the example selector
    example_selector = get_example_selector()
    
    # Use the `select_examples` method to get the best matching examples for a given input
    best_matching_examples = example_selector.select_examples({"input": input_text})
    
    # Return the query from the best matching example
    if best_matching_examples:
        return best_matching_examples[0]['query']
    else:
        return "No similar query found."

# Input question
input_question = "Show me all customers with a current account."

# Get the most similar query
similar_query = get_similar_query(input_question)

# Print the result
print(f"Input question: {input_question}")
print(f"Most similar SQL query: {similar_query}")

import pydantic
print(pydantic.__version__)
