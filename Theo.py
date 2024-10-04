import os
from dotenv import load_dotenv
from sqlalchemy import inspect
import pandas as pd
import openai
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

# Load environment variables
load_dotenv()

db_user = os.getenv("db_user")
db_password = os.getenv("db_password")
db_host = os.getenv("db_host")
db_name = os.getenv("db_name")
db_port = os.getenv('db_port')

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define database connection using SQLDatabase
tables_to_include = ['V_ACCOUNT', 'V_CUSTOMER']  # You can modify to include other tables
db = SQLDatabase.from_uri(
    f"mssql+pyodbc://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?driver=ODBC+Driver+17+for+SQL+Server",
    include_tables=tables_to_include, view_support=True
)

# Create an inspector object to retrieve table/column info
inspector = inspect(db._engine)

# Function to call OpenAI to generate table and column descriptions
def generate_description(prompt):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.5
    )
    print(response.choices[0].message.content.strip()
)
    return response.choices[0].message.content.strip()


# Initialize a list to store the final results
schema_info = []

# Loop through all the tables
tables = inspector.get_table_names()
for table in tables:
    # Step 1: Generate a description for the table
    table_description_prompt = f"Describe the purpose and contents of the table named '{table}' in a relational database."
    table_description = generate_description(table_description_prompt)

    # Step 2: Get columns and generate descriptions for each
    columns = inspector.get_columns(table)
    for col in columns:
        column_name = col['name']
        column_type = str(col['type'])

        # Step 3: Generate a description for each column
        column_description_prompt = f"Describe the purpose and meaning of the column '{column_name}' of type '{column_type}' in the table '{table}'."
        column_description = generate_description(column_description_prompt)

        # Append the table and column descriptions to the schema_info list
        schema_info.append({
            "TABLENAME": table,
            "TABLE_DESCRIPTION": table_description,
            "COLUMN_NAME": column_name,
            "COLUMN_TYPE": column_type,
            "COLUMN_DESCRIPTION": column_description
        })

# Step 4: Convert the schema_info list into a pandas DataFrame
schema_df = pd.DataFrame(schema_info)

# Step 5: Write the DataFrame to a CSV file
csv_file_path = "table_column_descriptions.csv"
schema_df.to_csv(csv_file_path, index=False)

print(f"Descriptions saved to {csv_file_path}")
