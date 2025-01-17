import streamlit as st
from examples import get_example_selector
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,FewShotChatMessagePromptTemplate,PromptTemplate


example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}\nSQLQuery:"),
        ("ai", "{query}"),
    ]
)

try:
    example_selector = get_example_selector()
except Exception as e:
    # Handle any other generic errors
    st.error("Something went wrong. Please contact datamanagement@bk.rw for support.")
    st.stop()

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    example_selector=example_selector,
    input_variables=["input", "top_k"],
)

# # Define a dictionary for definitions
# definitions = {
#     "Customer": "A customer is defined as an entity holding an account  with CATEGORY starting with digit of  '1' or '6', excluding any account within categories '1080' or '1031'.",
#     "Current Account": "Accounts with with a CATEGORY starting with digit '1', excluding category '1080' or '1031'. Predominantly used for routine transactions.",
#     "Loan Account": "Accounts with with a CATEGORY starting with digit '3', specifically associated with loan purposes.",
#     "VIP Customer": "A high-value status customer identified by target of value VIP or VVIP.",
#     "Transaction Date": "The date of the most recent transaction on the aacount (LAST_TRANS_DATE)",
#     "Active Account": "A current account (non-1080/1031) with at least one transaction in the last 90 days, or a savings account with at least one transaction in the last 360 days.",
#     "Inactive Account": "A current account (non-1080/1031) without transactions for over 90 days but with transaction history within the last 180 days, or a savings account with no transactions in the last 360 days.",
#     "Dormant Account": "A current account (non-1080/1031) inactive for more than 180 days but with transaction history within the past 360 days.",
#     "DomClosed Account": "A current account (non-1080/1031) inactive for over 360 days but with transactions in the preceding 1800 days.",
#     "Unclaimed Account": "A current account (non-1080/1031) with no transactions for at least 1800 days.",
#     "Active Customer": "A customer characterized by ownership of at least one active current or savings account.",
#     "Inactive Customer": "A customer with no active accounts but possessing at least one inactive current or savings account.",
#     "Dormant Customer": "A customer without active or inactive accounts, owning at least one dormant account.",
#     "DomClosed Customer": "A customer with no active, inactive, or dormant accounts, but maintaining at least one dom closed account.",
#     "Unclaimed Customer": "A customer whose only accounts are unclaimed current accounts, with no active, inactive, dormant, or dom closed accounts present",
#     "Churn Customer": "A customer who has only DomClosed Accounts or Unclaimed Accounts and no other accounts.indicating the likelihood of service termination or abandonment.",
#     "Churn Rate": " it is the percentage of Churn Customers over total number of customers (according to our definition).",
#     "EMPLOYMENT STATUS": "Whether a customer is employed, unemployed, retired, students, etc, determined using EMPLOYMENT_STATUS IN CUSTOMER TABLE",
#     "BK Card": "A BK Card is defined by specific card number prefixes, which include ISO_TRUN_PAN values of ('446999', '471375', '471376', '471377', '512952', '517315', '526111', '532018', '534617', '513904')."
# }

# # Convert the definitions dictionary to a string format
# definitions_string = "\n".join([f"- {key}: {value}" for key, value in definitions.items()])

# Construct the final prompt using the entire dictionary as a string
final_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """You are a MSSQL expert. Given an input question, create a syntactically correct SQL (MSSQL) query to run unless otherwise specified.

     Here is the relevant table info: {table_info}

     Below are a number of examples of questions and their corresponding SQL queries. These examples are correct and should be trusted. 
     Additionally, the SQL comments included with each example should be taken into account when designing the MS SQL query.

     Here are the relevant definitions for this query:
     {context}

     IMPORTANT: When generating SQL queries, make sure to apply the definitions provided above. If a definition isn't provided in the context, use standard business logic.
     """),
    few_shot_prompt,
    MessagesPlaceholder(variable_name="messages"),
    ("human", "{standalone_question}"),
])

input_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an intelligent assistant that provides data insights to Bank of Kigali, created by a team of Data Scientists and Engineers from the Datamanagement Department at Bank of Kigali. Your task is to determine whether a given question is a general inquiry, a business process question, or a data-related request. If the question is unclear or ambiguous, you must ask a follow-up question to clarify the intent before proceeding."),

        ("human", """1. If the question is a general inquiry, such as greetings (e.g., "hello", "hi", "how are you?", "what time is it?", who created you? etc.), answer the question conversationally. Keep your response friendly and professional.
   
    2. If the question is about the Bank of Kigali business process, use the definitions to provide a clear answer wherever it is applicable.

    3. If the question involves retrieving data, running a query, or anything technical involving a table, dataset, or database, respond with the number "1".

    4. If the question is unclear or too vague to categorize, ask a clarifying follow-up question. For example, if the user asks "Tell me about customers," you should respond with, "Could you clarify if you're asking about a specific type of customer ,customer activity, or if you're looking for data on customers?"

    Examples of general inquiries:
    - "Hello, how are you?"
    - "What time is it?"
    - "Tell me a joke."
    - "Good morning!"
    - "Who are you?"

    Examples of business process questions:
    - "What is a dormant customer?" 
    - "How do you identify VIP customers?"
    - "How do you calculate churn rate?"
    - "What is an active active account?"

    Examples of data-related requests:
    - "Show me the list of accounts that transacted last week."
    - "Retrieve the customer info from the database."
    - "How many new customers registered last month?"
    - "How many inactive customers do we have?"
    - "What is the current churn rate?"
    """),

        ("human", "Here is the chat history:"),
        MessagesPlaceholder(variable_name="messages"),  # Dynamically includes the chat history

        ("human", "Here is the question: \"{standalone_question}\""),
        ("human", "Below are the official Bank of Kigali business definitions:"),

        ("human", "{context}"),

        ("human", """Your response should either:
        - Be a conversational answer if it’s a general inquiry,
        - Provide a clear business explanation using the definitions if it’s a business process question,
        - Return "1" if it’s a data-related request,
        - Or ask a follow-up question to clarify the intent if the question is unclear.""")
    ]
)




answer_prompt = PromptTemplate.from_template(
   """
    Given the following user question, corresponding MSSQL query, SQL result summary, and the list of column names, 
    answer the user question in a professional, human-like manner. Handle cases when the result summary is empty or contains errors.
    Additionally, determine whether a chart is needed (e.g., bar, pie, line, area, scatter, histogram, box, funnel) based on the user's question and the result summary. 
    If a chart is needed, return the columns to be used for the chart in a list format.
    
    In case the SQL query and result summary are valid and contain no errors, provide a brief explanation of how the answer was derived 
    based on the query logic and the information retrieved from the database without mentioning table names.
    
    Return the result in JSON format as follows:
    
    {{
        "Answer": "<Provide a human-readable and professional answer>",
        "chart_type": "<Determine if a chart is needed, and if so, specify the type (e.g., bar, pie, line, area, scatter, histogram, box, funnel)>",
        "Column_names": "<List of columns to use if a chart is needed>",
    }}
    
    Question: {standalone_question}
    SQL Query: {query}
    Result Summary: {Summary}
    Column Names: {data_column}
   """
)

check_query_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert SQL assistant tasked with validating SQL queries against business definitions and requirements. Please assess the given query to ensure it matches the specified definitions and the original question being asked. Relevant table details are available here: {table_info}"""),

    ("human", "When generating or evaluating any SQL query, strictly apply the definitions provided. For clarity, if a question seeks the number of retail customers, the definition specifies a customer as having an account with a category starting with 1 or 6, excluding category 1080."),

    ("human", "Below are the official business definitions that must be strictly followed:"),
    ("human", "{context}"),  # Include business definitions dynamically

    ("human", "The original inquiry was as follows: {question}"),  # Explicitly include the original question

    ("human", "The SQL query that was generated based on the above question is as follows:"),
    ("human", "{query}"),  # Present the generated query

    ("human", "Here is the chat history:"),
    MessagesPlaceholder(variable_name="messages"),  # Include the conversation history

    ("human", "Your task is to review the SQL query. Please do one of the following:\n- Return the original query if it fully complies with the given definitions,\n- Provide a corrected SQL query that fully aligns with the definitions. Do not include any explanations or extra information in your response.")
])

condense_question_prompt = PromptTemplate(
    input_variables=['chat_history', 'question'], 
    template="""Given the following conversation and a follow up question (at the end), 
rephrase the follow up question to be a standalone question, in the same language as the follow up question.\n\n
Chat History:\n{chat_history}\n
Keep in mind that the last human message in chat history is the same as the follow up question.\n
Follow up question: {question}\n
Standalone question:"""
)

def answer_template(language="english"):
    """Pass the standalone question along with the chat history and context 
    to the `LLM` wihch will answer"""
    
    template = f"""Answer the question at the end, using only the following context (delimited by <context></context>).
Your answer must be in the language at the end. 

<context>
{{chat_history}}

{{context}} 
</context>

Question: {{question}}

Language: {language}.
"""
    return template