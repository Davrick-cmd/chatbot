
from examples import get_example_selector
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,FewShotChatMessagePromptTemplate,PromptTemplate

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}\nSQLQuery:"),
        ("ai", "{query}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    example_selector=get_example_selector(),
    input_variables=["input","top_k"],
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a MSSQL expert. Given an input question, create a syntactically correct SQL (MSSQL) query to run. Unless otherwise specificed.\n\nHere is the relevant table info: {table_info}\n\nBelow are a number of examples of questions and their corresponding SQL queries."),
        few_shot_prompt,
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input}"),
    ]
)


answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding MSSQL query, and SQL result summary, answer the user question as a human handle cases when resuly summary is empty as errors.

Question: {question}
SQL Query: {query}
Result Summary: {Summary}
Answer: """
)

input_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an intelligent assistant that provides data insights to Bank of Kigali created by a team of Data Scientists and Engineers at Bank of Kigali as part of data management initiatives. Your task is to determine whether a given question is a general inquiry or a data-related request."),

        ("human", """1. If the question is a general inquiry, such as greetings (e.g., "hello", "hi", "how are you?", "what time is it?", etc.), or casual conversation, answer the question as a human would. Keep your response friendly and professional.
   
2. If the question is related to retrieving data, running a query, or anything involving a specific table, dataset, or technical process, respond only with the number 1.

Examples of general inquiries:
- "Hello, how are you?"
- "What time is it?"
- "Hi, tell me a joke."
- "Good morning!"

Examples of data-related requests:
- "Show me the sales data for Q3."
- "Retrieve the customer info from the database."
- "Run a query on the accounts table."
- "How many users registered last month?"
"""),

        ("human", "Here is the chat history:"),
        MessagesPlaceholder(variable_name="messages"),  # This will dynamically include the chat history

        ("human", "Here is the question: \"{question}\""),

        ("human", "Your response should either:\n- Be a conversational answer if it’s a general inquiry, or\n- Return \"1\" if it’s data-related.")
    ]
)


