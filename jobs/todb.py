import os
import json
from sqlalchemy import create_engine, Table, Column, Integer, String, Text, MetaData, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError


DB_USER = "ADS_USER2"
DB_PASSWORD = "Pass!123$"
DB_HOST =  "10.24.34.157"
DB_PORT = 1433
DB_NAME = "ANALYTICS_BI_DB"



print(DB_USER)
print(DB_PASSWORD)
print(DB_HOST)
print(DB_PORT)
print(DB_NAME)

# Function to parse the log file and extract JSON objects
def parse_log_file(file_path):
    log_entries = []
    with open(file_path, 'r') as log_file:
        for line in log_file:
            try:
                # Extract JSON part after the first ' - '
                _, json_part = line.split(" - ", 1)
                log_entry = json.loads(json_part.strip())
                log_entries.append(log_entry)
            except (ValueError, json.JSONDecodeError) as e:
                print(f"Error parsing line: {line.strip()} | Error: {e}")
    return log_entries

# Function to save log entries to an SQL database
def save_logs_to_db(log_entries):


    # Construct connection string using the defined variables
    connection_string = (
        f"mssql+pyodbc://{DB_USER}:{DB_PASSWORD}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        "?driver=ODBC+Driver+17+for+SQL+Server"
    )

    engine = create_engine(connection_string)
    Session = sessionmaker(bind=engine)
    session = Session()
    metadata = MetaData()

    # Define the Logs table
    logs_table = Table(
        "conversation_logs", metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("timestamp", String(50)),
        Column("user_id", String(50)),
        Column("question", Text),
        Column("sql_query", Text),
        Column("answer", Text),
        Column("chart_type", String(50)),
        Column("feedback", String(50)),
        Column("feedback_comment", Text)
    )

    try:
        # Ensure the Logs table exists
        metadata.create_all(engine)

        # Insert each log entry into the database
        for entry in log_entries:
            try:
                # Format the timestamp to a compatible format
                formatted_timestamp = entry.get("timestamp").replace("T", " ")[:19]  # 'YYYY-MM-DD HH:MM:SS'
                
                session.execute(logs_table.insert().values(
                    timestamp=formatted_timestamp,
                    user_id=entry.get("user_id"),
                    question=entry.get("question"),
                    sql_query=entry.get("sql_query"),
                    answer=entry.get("answer"),
                    chart_type=entry.get("chart_type"),
                    feedback=entry.get("feedback") if entry.get("feedback") is not None else None,
                    feedback_comment=entry.get("feedback_comment") if entry.get("feedback_comment") is not None else None
                ))
            except SQLAlchemyError as e:
                print(f"Error inserting entry into database: {entry} | Error: {e}")

        session.commit()
        print("Logs saved to the database successfully!")
    except SQLAlchemyError as e:
        print(f"Error saving logs to the database: {str(e)}")
        session.rollback()
    finally:
        session.close()

# Main script
if __name__ == "__main__":
    # Path to the .log file
    log_file_path = "app/logs/conversation.log"

    # Parse log file and save to the database
    log_entries = parse_log_file(log_file_path)
    if log_entries:
        save_logs_to_db(log_entries)
    else:
        print("No valid log entries found.")
