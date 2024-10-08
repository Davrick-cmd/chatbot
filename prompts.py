
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

# Define a dictionary for definitions
definitions = {
    "Customer": "A customer is considered to have an account with a category starting with 1 or 6 and should not have category 1080 or 1031.",
    "Current Account": "Accounts that have a category starting with '1' and should not have category 1080, typically used for day-to-day transactions.",
    "Loan Account": "Accounts that have a category starting with '3', typically used for loans.",
    "VIP Customer": "Customers who have a target of 57, 66, or 91, indicating high-value status.",
    "Transaction Date": "The most recent debit or credit by the bank or the customer (Maximum(DATE_LAST_DR_BANK, DATE_LAST_CR_BANK, DATE_LAST_DR_CUST, DATE_LAST_CR_CUST))",
    "Active Account": "A current account that has had at least one transaction (transaction date >= current date - 90 days) in the last 90 days, or a savings account that has had at least one transaction (transaction date >= current date - 720 days) in the last 720 days.",
    "Inactive Account": "A current account that hasn't transacted in the last 90 days (transaction date < current date - 90 days) but has transacted at least once in the last 180 days (transaction date >= current date - 180 days), or a savings account that hasn't transacted in the last 720 days (transaction date < current date - 720 days).",
    "Dormant Account": "A current account that hasn't transacted in the last 180 days but has had at least one transaction in the last 360 days. (current date - 90 < = transaction date >= current date - 180 days)",
    "Dom Closed Account": "A current account that hasn't transacted in the last 360 days (1 year) but has had at least one transaction in the last 1800 days (5 years).",
    "Unclaimed Account":  "A current account that hasn't transacted at least oncee in the last 1800 days",
    "Active Customer":"A customer with at least one active account.",
    "Inactive Customer":"A customer with no active account but has at least one inactive account",
    "Dormant Customer": "A customer with no active or inactive account but has at least one dormant account",
    "Dom closed Customer": "A customer with no active or inactive or dormant account but has at least one dom close account",
    "Unclaimed Customer": "A customer with no other accounts other than unclaimed account",
    "Churn customer": "A churn customer is a dom closed customer or unclaimed customer"
}

# Convert the definitions dictionary to a string format
definitions_string = "\n".join([f"- {key}: {value}" for key, value in definitions.items()])

# Construct the final prompt using the entire dictionary as a string
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         f"""You are a MSSQL expert. Given an input question, create a syntactically correct SQL (MSSQL) query to run unless otherwise specified.

         Here is the relevant table info: {{table_info}}

         Below are a number of examples of questions and their corresponding SQL queries. These examples are correct and should be trusted. 
         Additionally, the SQL comments included with each example should be taken into account when designing the MS SQL query.

         Definitions:
         {definitions_string}

         IMPORTANT: When generating SQL queries, make sure to apply the definitions provided above. For example, if a question asks for the number of retail customers, remember that a customer is defined as having an account with a category starting with 1 or 6 and not having category 1080.
         """
        ),
        few_shot_prompt,
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input}"),
    ]
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


answer_prompt = PromptTemplate.from_template(
   """
    Given the following user question, corresponding MSSQL query, SQL result summary, and the list of column names, 
    answer the user question in a professional, human-like manner. Handle cases when the result summary is empty or contains errors.
    Additionally, determine whether a chart is needed (e.g., bar, pie, line, area, scatter, histogram, box, funnel) based on the user's question and the result summary. 
    If a chart is needed, return the columns to be used for the chart in a list format.
    
    Return the result in JSON format as follows:
    
    {{
        "Answer": "<Provide a human-readable and professional answer>",
        "chart_type": "<Determine if a chart is needed, and if so, specify the type (e.g., bar, pie, line, area, scatter, histogram, box, funnel)>",
        "Column_names": "<List of columns to use if a chart is needed>"
    }}
    
    Question: {question}
    SQL Query: {query}
    Result Summary: {Summary}
    Column Names: {data_column}
    """
)

