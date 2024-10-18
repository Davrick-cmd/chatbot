import json

examples = [
    {
        "input": "How many customers do we have?",
        "query": """-- A customer is defined as someone who has an account with a category starting with '1' (e.g., Current) or '6' (e.g., Savings), 
                    -- and should not have an account in category '1080' (e.g., Suspended Accounts).
                SELECT COUNT(DISTINCT c.CUSTOMER_NO)
                FROM BOT_CUSTOMER c
                JOIN ChatbotAccounts a
                    ON c.CUSTOMER_NO = a.CUSTOMER_NUMBER
                WHERE a.CATEGORY NOT IN ('1080', '1031')
                AND SUBSTRING(a.CATEGORY, 1, 1) IN ('1', '6');
                """
    },
    {
        "input": "How many accounts transacted yesterday?",
        "query": """SELECT COUNT(DISTINCT ACCOUNT_NUMBER) AS account_count 
                    FROM ChatbotAccounts 
                    WHERE LAST_TRANS_DATE = CAST(GETDATE() - 1 AS DATE);"""
    },
    {
        "input":"How many active account do we have?",
        "query" : """SELECT COUNT(DISTINCT ACCOUNT_NUMBER)
                    FROM ChatbotAccounts
                    WHERE
                        (CATEGORY LIKE '1%' AND CATEGORY NOT IN ('1080', '1031') AND LAST_TRANS_DATE >= DATEADD(DAY, -90, GETDATE())) -- Active for current accounts
                        OR
                        (CATEGORY LIKE '6%' AND LAST_TRANS_DATE >= DATEADD(DAY, -360, GETDATE())) -- Active for savings accounts
                        AND CATEGORY NOT IN ('1080', '1031') --Unsecured Financing Accounts
                    """

    },
    {   "input":"How many Inactive accounts?",
        "query":"""SELECT COUNT(DISTINCT ACCOUNT_NUMBER)
                FROM ChatbotAccounts
                WHERE 
                    (CATEGORY LIKE '1%' AND  CATEGORY NOT IN ('1080', '1031') AND LAST_TRANS_DATE < DATEADD(DAY, -90, GETDATE())  -- Inactive for current accounts
                    AND LAST_TRANS_DATE >= DATEADD(DAY, -180, GETDATE()))  -- At least one transaction in last 180 days
                    OR
                    (CATEGORY LIKE '6%' AND  LAST_TRANS_DATE < DATEADD(DAY, -360, GETDATE()));  -- Savings accounts without transactions in last 360 days
                """

    },
    {
        "input": "How many customers are in the retail segment?",
        "query": """-- Segment ('RETAIL','AGRICULTURE','CORPORATE','INSTITUTIONAL BANKING','SME') 
                        SELECT COUNT(DISTINCT c.CUSTOMER_NO)
                        FROM BOT_CUSTOMER c
                        JOIN ChatbotAccounts a
                            ON c.CUSTOMER_NO = a.CUSTOMER_NUMBER
                        WHERE c.SEGMENT like = 'RETAIL'
                        AND a.CATEGORY NOT IN ('1080', '1031')
                        AND SUBSTRING(a.CATEGORY, 1, 1) IN ('1', '6');
                    """  # Fixed missing closing parenthesis
    },
    {
        "input": "How many customers are in the SME segment?",
        "query": """-- Segment '2' corresponds to SME (Small and Medium Enterprises) customers.
                        SELECT COUNT(DISTINCT c.CUSTOMER_NO)
                        FROM BOT_CUSTOMER c
                        JOIN ChatbotAccounts a
                            ON c.CUSTOMER_NO = a.CUSTOMER_NUMBER
                        WHERE c.SEGMENT = 'SME'
                        AND a.CATEGORY NOT IN ('1080', '1031')
                        AND SUBSTRING(a.CATEGORY, 1, 1) IN ('1', '6');"""  # Fixed missing closing parenthesis
    },
    {
        "input": "How many Loan accounts?",
        "query": """-- Categories starting with '3' correspond to loan accounts.
                    SELECT COUNT(DISTINCT ACCOUNT_NUMBER) 
                    FROM ChatbotAccounts 
                    WHERE CATEGORY LIKE '3%';"""
    },
    {
        "input": "How many Current accounts?",
        "query": """-- Categories starting with '1' correspond to current accounts.
                    SELECT COUNT(DISTINCT ACCOUNT_NUMBER) 
                    FROM ChatbotAccounts 
                    WHERE (CATEGORY LIKE '1%' AND CATEGORY NOT IN ('1080', '1031'));"""
    },
    {
        "input": "How many Savings accounts?",
        "query": """-- Categories starting with '6' correspond to savings accounts.
                    SELECT COUNT(DISTINCT ACCOUNT_NUMBER) 
                    FROM ChatbotAccounts 
                    WHERE CATEGORY LIKE '6%'
                ;"""
    },
    {
        "input": "Give me account activity status",
        "query": """
                WITH Filtered_Accounts AS (
                    SELECT 
                        ACCOUNT_NUMBER,  -- Ensure this column represents your account identifier
                        CATEGORY,
                        LAST_TRANS_DATE,
                        CASE 
                            WHEN ((CATEGORY LIKE '1%' AND LAST_TRANS_DATE > DATEADD(DAY, -90, GETDATE())) 
                                OR (CATEGORY LIKE '6%' AND LAST_TRANS_DATE > DATEADD(DAY, -360, GETDATE()))) THEN 'ACTIVE'
                            WHEN (CATEGORY LIKE '1%' AND LAST_TRANS_DATE <= DATEADD(DAY, -90, GETDATE()) 
                                AND LAST_TRANS_DATE > DATEADD(DAY, -180, GETDATE())) 
                                OR (CATEGORY LIKE '6%' AND LAST_TRANS_DATE <= DATEADD(DAY, -360, GETDATE())) THEN 'INACTIVE'
                            WHEN (CATEGORY LIKE '1%' AND LAST_TRANS_DATE <= DATEADD(DAY, -180, GETDATE()) 
                                AND LAST_TRANS_DATE > DATEADD(DAY, -360, GETDATE())) THEN 'DORMANT'
                            WHEN (CATEGORY LIKE '1%' AND LAST_TRANS_DATE <= DATEADD(DAY, -360, GETDATE()) 
                                AND LAST_TRANS_DATE > DATEADD(DAY, -1800, GETDATE())) THEN 'DOMCLOSED'
                            WHEN (CATEGORY LIKE '1%' AND LAST_TRANS_DATE < DATEADD(DAY, -1800, GETDATE())) THEN 'UNCLAIMED'
                            ELSE NULL
                        END AS Status,
                        CASE 
                            WHEN SUBSTRING(CATEGORY, 1, 1) = '1' THEN 'Current Account'
                            WHEN SUBSTRING(CATEGORY, 1, 1) = '3' THEN 'Loan Account'
                            WHEN SUBSTRING(CATEGORY, 1, 1) = '6' THEN 'Savings Account'
                            ELSE 'Other Accounts'
                        END AS Account_Type
                    FROM ChatbotAccounts
                    WHERE CATEGORY NOT IN ('1080', '1031')
                )

                SELECT 
                    Account_Type,
                    COUNT(DISTINCT ACCOUNT_NUMBER) AS Total_Accounts,
                    SUM(CASE WHEN Status = 'ACTIVE' THEN 1 ELSE 0 END) AS Active_Accounts,
                    SUM(CASE WHEN Status = 'INACTIVE' THEN 1 ELSE 0 END) AS Inactive_Accounts,
                    SUM(CASE WHEN Status = 'DORMANT' THEN 1 ELSE 0 END) AS Dormant_Accounts,
                    SUM(CASE WHEN Status = 'DOMCLOSED' THEN 1 ELSE 0 END) AS Dom_Closed_Accounts,
                    SUM(CASE WHEN Status = 'UNCLAIMED' THEN 1 ELSE 0 END) AS Unclaimed_Accounts,
                    CAST(
                        (SUM(CASE WHEN Status IN ('DOMCLOSED', 'UNCLAIMED') THEN 1 ELSE 0 END) * 1.0) / 
                        NULLIF(COUNT(DISTINCT ACCOUNT_NUMBER), 0) * 100 AS DECIMAL(10, 2)
                    ) AS Churn_Rate
                FROM Filtered_Accounts 
                GROUP BY Account_Type

                """
    },
    { 
        "input": "Give me a list of 10 corporate customers names?",
        "query": """-- Corporate customers are those in segment '4'.
                    SELECT TOP 10 c.SHORT_NAME
                    FROM BOT_CUSTOMER c
                    JOIN ChatbotAccounts a
                        ON c.CUSTOMER_NO = a.CUSTOMER_NUMBER
                    WHERE c.SEGMENT = '4'
                    AND a.CATEGORY <> '1080'
                    AND SUBSTRING(a.CATEGORY, 1, 1) IN ('1', '6')
                ;""" 
    },
    {
        "input": "Give me a list of all BK STAFF customers?",
        "query": """-- TARGET ('BK STAFF','CEOs','DIASPORA','POLITICAL EXPOSED PERSONS','STUDENTS',etc).
                    SELECT * 
                    FROM BOT_CUSTOMER 
                    WHERE TARGET = 'BK STAFF'
                    ;"""
    },
    {
        "input": "How many student customers do we have?",
        "query": """
                    SELECT COUNT(DISTINCT CUSTOMER_NO) 
                    FROM BOT_CUSTOMER 
                    WHERE TARGET = 'STUDENT';"""
    },
    { 
        "input": "Give a list of all targets to identify different customers",
        "query": "SELECT DISTINCT TARGET FROM BOT_CUSTOMER;"
    },
    {
        "input":"How many inactive customers do we have?",
        "query":"""WITH Filtered_Accounts AS (
                    SELECT 
                        CUSTOMER_NUMBER,
                        CASE 
                            WHEN ((CATEGORY LIKE '1%' AND LAST_TRANS_DATE > DATEADD(DAY, -90, GETDATE())) 
                                OR (CATEGORY LIKE '6%' AND LAST_TRANS_DATE > DATEADD(DAY, -360,GETDATE()))) THEN 1
                            WHEN CATEGORY LIKE '1%' AND LAST_TRANS_DATE <= DATEADD(DAY, -90, GETDATE()) 
                                AND LAST_TRANS_DATE > DATEADD(DAY, -180, GETDATE()) 
                                OR CATEGORY LIKE '6%' AND LAST_TRANS_DATE <= DATEADD(DAY, -360,GETDATE())THEN 2
                            WHEN CATEGORY LIKE '1%' AND LAST_TRANS_DATE <= DATEADD(DAY, -180, GETDATE()) 
                                AND LAST_TRANS_DATE > DATEADD(DAY, -360, GETDATE()) THEN 3
                            WHEN CATEGORY LIKE '1%' AND LAST_TRANS_DATE <= DATEADD(DAY, -360, GETDATE()) 
                                AND LAST_TRANS_DATE > DATEADD(DAY, -1800, GETDATE()) THEN 4
                            WHEN CATEGORY LIKE '1%' AND LAST_TRANS_DATE < DATEADD(DAY, -1800, GETDATE()) THEN 5
                            ELSE NULL
                        END AS Status
                    FROM ChatbotAccounts
                    WHERE CATEGORY NOT IN ('1080', '1031')
                ),
                
                Customer_Status AS (
                    SELECT 
                        CUSTOMER_NUMBER,
                        MIN(status) AS Status_Rank
                    FROM Filtered_Accounts
                    WHERE Status IS NOT NULL
                    GROUP BY CUSTOMER_NUMBER
                )
                
                
                SELECT
                    COUNT(DISTINCT c.CUSTOMER_NO) AS TOTAL_CUSTOMERS,
                    COUNT(DISTINCT CASE WHEN cs.Status_Rank = 1 THEN c.CUSTOMER_NO END) AS ACTIVE_CUSTOMERS,
                    COUNT(DISTINCT CASE WHEN cs.Status_Rank = 2 THEN c.CUSTOMER_NO END) AS INACTIVE_CUSTOMERS,
                    COUNT(DISTINCT CASE WHEN cs.Status_Rank = 3 THEN c.CUSTOMER_NO END) AS DORMANT_CUSTOMERS,
                    COUNT(DISTINCT CASE WHEN cs.Status_Rank = 4 THEN c.CUSTOMER_NO END) AS DOM_CLOSED_CUSTOMERS,
                    COUNT(DISTINCT CASE WHEN cs.Status_Rank = 5 THEN c.CUSTOMER_NO END) AS UNCLAIMED_CUSTOMERS,
                    CAST(
                    (COUNT(DISTINCT CASE WHEN cs.Status_Rank = 4 OR cs.Status_Rank = 5 THEN c.CUSTOMER_NO END) *1.0) /
                    (COUNT(DISTINCT c.CUSTOMER_NO)) * 100 AS DECIMAL(10,2)) AS CHURN_RATE

                FROM BOT_CUSTOMER c
                JOIN Customer_Status cs ON c.CUSTOMER_NO = cs.CUSTOMER_NUMBER;
                    """
    },
    {   "input":"How many dom closed customers do we have?",
        "query":"""
                WITH Filtered_Accounts AS (
                    SELECT 
                        CUSTOMER_NUMBER,
                        CASE 
                            WHEN ((CATEGORY LIKE '1%' AND LAST_TRANS_DATE > DATEADD(DAY, -90, GETDATE())) 
                                OR (CATEGORY LIKE '6%' AND LAST_TRANS_DATE > DATEADD(DAY, -360,GETDATE()))) THEN 1
                            WHEN CATEGORY LIKE '1%' AND LAST_TRANS_DATE <= DATEADD(DAY, -90, GETDATE()) 
                                AND LAST_TRANS_DATE > DATEADD(DAY, -180, GETDATE()) 
                                OR CATEGORY LIKE '6%' AND LAST_TRANS_DATE <= DATEADD(DAY, -360,GETDATE())THEN 2
                            WHEN CATEGORY LIKE '1%' AND LAST_TRANS_DATE <= DATEADD(DAY, -180, GETDATE()) 
                                AND LAST_TRANS_DATE > DATEADD(DAY, -360, GETDATE()) THEN 3
                            WHEN CATEGORY LIKE '1%' AND LAST_TRANS_DATE <= DATEADD(DAY, -360, GETDATE()) 
                                AND LAST_TRANS_DATE > DATEADD(DAY, -1800, GETDATE()) THEN 4
                            WHEN CATEGORY LIKE '1%' AND LAST_TRANS_DATE < DATEADD(DAY, -1800, GETDATE()) THEN 5
                            ELSE NULL
                        END AS Status
                    FROM ChatbotAccounts
                    WHERE CATEGORY NOT IN ('1080', '1031')
                ),
                
                Customer_Status AS (
                    SELECT 
                        CUSTOMER_NUMBER,
                        MIN(status) AS Status_Rank
                    FROM Filtered_Accounts
                    WHERE Status IS NOT NULL
                    GROUP BY CUSTOMER_NUMBER
                )
                
                
                SELECT
                    COUNT(DISTINCT c.CUSTOMER_NO) AS TOTAL_CUSTOMERS,
                    COUNT(DISTINCT CASE WHEN cs.Status_Rank = 1 THEN c.CUSTOMER_NO END) AS ACTIVE_CUSTOMERS,
                    COUNT(DISTINCT CASE WHEN cs.Status_Rank = 2 THEN c.CUSTOMER_NO END) AS INACTIVE_CUSTOMERS,
                    COUNT(DISTINCT CASE WHEN cs.Status_Rank = 3 THEN c.CUSTOMER_NO END) AS DORMANT_CUSTOMERS,
                    COUNT(DISTINCT CASE WHEN cs.Status_Rank = 4 THEN c.CUSTOMER_NO END) AS DOM_CLOSED_CUSTOMERS,
                    COUNT(DISTINCT CASE WHEN cs.Status_Rank = 5 THEN c.CUSTOMER_NO END) AS UNCLAIMED_CUSTOMERS,
                    CAST(
                    (COUNT(DISTINCT CASE WHEN cs.Status_Rank = 4 OR cs.Status_Rank = 5 THEN c.CUSTOMER_NO END) *1.0) /
                    (COUNT(DISTINCT c.CUSTOMER_NO)) * 100 AS DECIMAL(10,2)) AS CHURN_RATE

                FROM BOT_CUSTOMER c
                JOIN Customer_Status cs ON c.CUSTOMER_NO = cs.CUSTOMER_NUMBER;
        """

    },
    {
        "input":"What is our churn rate number?",
        "query":"""
                WITH Filtered_Accounts AS (
                    SELECT 
                        CUSTOMER_NUMBER,
                        CASE 
                            WHEN ((CATEGORY LIKE '1%' AND LAST_TRANS_DATE > DATEADD(DAY, -90, GETDATE())) 
                                OR (CATEGORY LIKE '6%' AND LAST_TRANS_DATE > DATEADD(DAY, -360,GETDATE()))) THEN 1
                            WHEN CATEGORY LIKE '1%' AND LAST_TRANS_DATE <= DATEADD(DAY, -90, GETDATE()) 
                                AND LAST_TRANS_DATE > DATEADD(DAY, -180, GETDATE()) 
                                OR CATEGORY LIKE '6%' AND LAST_TRANS_DATE <= DATEADD(DAY, -360,GETDATE())THEN 2
                            WHEN CATEGORY LIKE '1%' AND LAST_TRANS_DATE <= DATEADD(DAY, -180, GETDATE()) 
                                AND LAST_TRANS_DATE > DATEADD(DAY, -360, GETDATE()) THEN 3
                            WHEN CATEGORY LIKE '1%' AND LAST_TRANS_DATE <= DATEADD(DAY, -360, GETDATE()) 
                                AND LAST_TRANS_DATE > DATEADD(DAY, -1800, GETDATE()) THEN 4
                            WHEN CATEGORY LIKE '1%' AND LAST_TRANS_DATE < DATEADD(DAY, -1800, GETDATE()) THEN 5
                            ELSE NULL
                        END AS Status
                    FROM ChatbotAccounts
                    WHERE CATEGORY NOT IN ('1080', '1031')
                ),
                
                Customer_Status AS (
                    SELECT 
                        CUSTOMER_NUMBER,
                        MIN(status) AS Status_Rank
                    FROM Filtered_Accounts
                    WHERE Status IS NOT NULL
                    GROUP BY CUSTOMER_NUMBER
                )
                
                SELECT 
                    COUNT(DISTINCT c.CUSTOMER_NO) AS TOTAL_CUSTOMERS,
                    COUNT(DISTINCT CASE WHEN cs.Status_Rank = 4 OR cs.Status_Rank = 5 THEN c.CUSTOMER_NO END) AS CHRN_CUSTOMERS,
                    CAST(
                    (COUNT(DISTINCT CASE WHEN cs.Status_Rank = 4 OR cs.Status_Rank = 5 THEN c.CUSTOMER_NO END) *1.0) /
                    (COUNT(DISTINCT c.CUSTOMER_NO)) * 100 AS DECIMAL(10,2)) AS CHURN_RATE

                FROM BOT_CUSTOMER c
                JOIN Customer_Status cs ON c.CUSTOMER_NO = cs.CUSTOMER_NUMBER;
                        """
    },
    {
        "input":"What is the current churn rate by segment",
        "query": """
                WITH Filtered_Accounts AS (
                    SELECT 
                        CUSTOMER_NUMBER,
                        CASE 
                            WHEN ((CATEGORY LIKE '1%' AND LAST_TRANS_DATE > DATEADD(DAY, -90, GETDATE())) 
                                OR (CATEGORY LIKE '6%' AND LAST_TRANS_DATE > DATEADD(DAY, -360,GETDATE()))) THEN 1
                            WHEN CATEGORY LIKE '1%' AND LAST_TRANS_DATE <= DATEADD(DAY, -90, GETDATE()) 
                                AND LAST_TRANS_DATE > DATEADD(DAY, -180, GETDATE()) 
                                OR CATEGORY LIKE '6%' AND LAST_TRANS_DATE <= DATEADD(DAY, -360,GETDATE())THEN 2
                            WHEN CATEGORY LIKE '1%' AND LAST_TRANS_DATE <= DATEADD(DAY, -180, GETDATE()) 
                                AND LAST_TRANS_DATE > DATEADD(DAY, -360, GETDATE()) THEN 3
                            WHEN CATEGORY LIKE '1%' AND LAST_TRANS_DATE <= DATEADD(DAY, -360, GETDATE()) 
                                AND LAST_TRANS_DATE > DATEADD(DAY, -1800, GETDATE()) THEN 4
                            WHEN CATEGORY LIKE '1%' AND LAST_TRANS_DATE < DATEADD(DAY, -1800, GETDATE()) THEN 5
                            ELSE NULL
                        END AS Status
                    FROM ChatbotAccounts
                    WHERE CATEGORY NOT IN ('1080', '1031')
                ),
                
                Customer_Status AS (
                    SELECT 
                        CUSTOMER_NUMBER,
                        MIN(status) AS Status_Rank
                    FROM Filtered_Accounts
                    WHERE Status IS NOT NULL
                    GROUP BY CUSTOMER_NUMBER
                )
                
                SELECT 
                    c.SEGMENT,
                    COUNT(DISTINCT c.CUSTOMER_NO) AS TOTAL_CUSTOMERS,
                    COUNT(DISTINCT CASE WHEN cs.Status_Rank = 1 THEN c.CUSTOMER_NO END) AS ACTIVE_CUSTOMERS,
                    COUNT(DISTINCT CASE WHEN cs.Status_Rank = 2 THEN c.CUSTOMER_NO END) AS INACTIVE_CUSTOMERS,
                    COUNT(DISTINCT CASE WHEN cs.Status_Rank = 3 THEN c.CUSTOMER_NO END) AS DORMANT_CUSTOMERS,
                    COUNT(DISTINCT CASE WHEN cs.Status_Rank = 4 THEN c.CUSTOMER_NO END) AS DOM_CLOSED_CUSTOMERS,
                    COUNT(DISTINCT CASE WHEN cs.Status_Rank = 5 THEN c.CUSTOMER_NO END) AS UNCLAIMED_CUSTOMERS,
                    CAST(
                    (COUNT(DISTINCT CASE WHEN cs.Status_Rank = 4 OR cs.Status_Rank = 5 THEN c.CUSTOMER_NO END) *1.0) /
                    (COUNT(DISTINCT c.CUSTOMER_NO)) * 100 AS DECIMAL(10,2)) AS CHURN_RATE

                FROM BOT_CUSTOMER c
                JOIN Customer_Status cs ON c.CUSTOMER_NO = cs.CUSTOMER_NUMBER
                GROUP BY c.SEGMENT;
                """
    },
    {
        "input": "Return the number of customers who joined the bank this year",
        "query":"""
                SELECT COUNT(DISTINCT c.CUSTOMER_NO)
                FROM BOT_CUSTOMER c
                JOIN ChatbotAccounts a
                    ON c.CUSTOMER_NO = a.CUSTOMER_NUMBER
                WHERE a.CATEGORY NOT IN ('1080', '1031')
                AND SUBSTRING(a.CATEGORY, 1, 1) IN ('1', '6')
				AND YEAR(c.JOINING_DATE) = YEAR(GETDATE());
                """
    },
    {
        "input":"Tell me about individual and non individual customer",
        "query":"""
                SELECT BK_TYPE_OF_CUST,COUNT(DISTINCT c.CUSTOMER_NO) AS NUMBER_CUSTOMERS
                FROM BOT_CUSTOMER c
                JOIN ChatbotAccounts a
                    ON c.CUSTOMER_NO = a.CUSTOMER_NUMBER
                WHERE a.CATEGORY NOT IN ('1080', '1031')
                AND SUBSTRING(a.CATEGORY, 1, 1) IN ('1', '6')
				GROUP BY BK_TYPE_OF_CUST;
            """
    },
    {
        "input":"Return the customers who transacted and those who never transacted",
        "query":"""
                WITH CustomerAccountStatus AS (
                    SELECT 
                        c.CUSTOMER_NO,
                        CASE 
                            WHEN a.LAST_TRANS_DATE = a.OPENING_DATE THEN 0  -- Never Transacted
                            ELSE 1  -- Has Transacted
                        END AS Transaction_Status
                    FROM 
                        BOT_CUSTOMER c
                    JOIN 
                        ChatbotAccounts a ON c.CUSTOMER_NO = a.CUSTOMER_NUMBER
                    WHERE 
                        a.CATEGORY NOT IN ('1080', '1031')  -- Exclude specific categories
                        AND SUBSTRING(a.CATEGORY, 1, 1) IN ('1', '6')  -- Include valid categories
                ),

                MaxTransactionStatus AS (
                    SELECT 
                        CUSTOMER_NO,
                        MAX(Transaction_Status) AS Max_Status
                    FROM 
                        CustomerAccountStatus
                    GROUP BY 
                        CUSTOMER_NO
                )

                SELECT 
                    SUM(CASE WHEN Max_Status = 0 THEN 1 ELSE 0 END) AS Customers_Never_Transacted,
                    SUM(CASE WHEN Max_Status = 1 THEN 1 ELSE 0 END) AS Customers_Has_Transacted
                FROM 
                    MaxTransactionStatus;
        """
    },
    {
        "input":"How many dormant customers will become dom closed customers in the 30 days if they don't transact",
        "query":"""
                WITH Filtered_Accounts AS (
                    SELECT 
                        CUSTOMER_NUMBER,
                        CASE 
                            WHEN ((CATEGORY LIKE '1%' AND LAST_TRANS_DATE > DATEADD(DAY, -90, GETDATE())) 
                                OR (CATEGORY LIKE '6%' AND LAST_TRANS_DATE > DATEADD(DAY, -360, GETDATE()))) THEN 1  -- Active
                            WHEN (CATEGORY LIKE '1%' AND LAST_TRANS_DATE <= DATEADD(DAY, -90, GETDATE()) 
                                AND LAST_TRANS_DATE > DATEADD(DAY, -180, GETDATE())) 
                                OR (CATEGORY LIKE '6%' AND LAST_TRANS_DATE <= DATEADD(DAY, -360, GETDATE())) THEN 2  -- Inactive
                            WHEN (CATEGORY LIKE '1%' AND LAST_TRANS_DATE <= DATEADD(DAY, -180, GETDATE()) 
                                AND LAST_TRANS_DATE > DATEADD(DAY, -360, GETDATE())) THEN 3  -- Dormant
                            WHEN (CATEGORY LIKE '1%' AND LAST_TRANS_DATE <= DATEADD(DAY, -360, GETDATE()) 
                                AND LAST_TRANS_DATE > DATEADD(DAY, -1800, GETDATE())) THEN 4  -- Dom Closed
                            WHEN (CATEGORY LIKE '1%' AND LAST_TRANS_DATE < DATEADD(DAY, -1800, GETDATE())) THEN 5  -- Unclaimed
                            ELSE NULL
                        END AS Status
                    FROM ChatbotAccounts
                    WHERE CATEGORY NOT IN ('1080', '1031')
                ),

                Customer_Status AS (
                    SELECT 
                        CUSTOMER_NUMBER,
                        MIN(Status) AS Status_Rank
                    FROM Filtered_Accounts
                    WHERE Status IS NOT NULL
                    GROUP BY CUSTOMER_NUMBER
                ),

                Dormant_Customers AS (
                    SELECT 
                        cs.CUSTOMER_NUMBER
                    FROM Customer_Status cs
                    WHERE cs.Status_Rank = 3  -- Identifying dormant customers
                )

                SELECT
                count(DISTINCT dc.CUSTOMER_NUMBER)
                FROM  
                    Dormant_Customers dc
                JOIN ChatbotAccounts a ON dc.CUSTOMER_NUMBER = a.CUSTOMER_NUMBER
                WHERE 
                    a.LAST_TRANS_DATE < DATEADD(DAY, -330, GETDATE());

        """
    }
]

def transform_to_jsonl(input_data_list, output_file, table_info_holder='', definitions_string=''):
    # Iterate over each input data entry
    for input_data in input_data_list:
        # Create the message structure for each entry
        output = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"You are a MSSQL expert. Given an input question, create a syntactically correct "
                        f"SQL (MSSQL) query to run unless otherwise specified. Here is the relevant table info: "
                        "{table_info_holder}. Definitions: {definitions_string}. IMPORTANT: When generating SQL queries, "
                        f"make sure to apply the definitions provided above."
                    )
                },
                {
                    "role": "user",
                    "content": input_data['input']
                },
                {
                    "role": "assistant",
                    "content": input_data['query']
                }
            ]
        }

        # Append the output to the specified JSONL file
        with open(output_file, 'a') as file:
            file.write(json.dumps(output) + '\n')


# Function to read examples from a JSON file
def load_examples_from_json(file_path):
    with open(file_path, 'r') as file:
        examples = json.load(file)
    return examples


# Function to convert list of dictionaries to JSON and save to a file
def save_to_json(data, output_file):
    # Preprocess each 'query' in the data to remove unnecessary spaces but keep \n intact
    for item in data:
        # Strip leading/trailing spaces from each line but keep \n
        item['query'] = "\n".join(line.strip() for line in item['query'].splitlines())

    # Write the JSON data to the output file
    with open(output_file, 'w') as file:
        file.write('[\n')  # Start the JSON array
        for i, item in enumerate(data):
            # Create a JSON string with the desired formatting
            formatted_json = json.dumps(item)  # No indent for compact representation
            file.write(f"  {formatted_json}")  # Indent each JSON object for clarity
            if i < len(data) - 1:
                file.write(',\n')  # Add a comma if not the last item
            else:
                file.write('\n')  # End with a newline for the last item
        file.write(']')  # Close the JSON array


# Specify the output JSON file
output_file_raw = 'examples_raw.json'


# Specify the path to your JSON file
input_json_raw_file = 'examples_raw.json'

# Load the examples from the JSON file



# Provide your table info and definitions string here
table_info = "{table_info}"
definitions_string = "{definitions_string}"

# Specify your output JSONL file
output_jsonl_file = 'utils/output.jsonl'


# Save the examples list to a JSON file

save_to_json(examples, output_file_raw)
print(f"Data saved to {output_file_raw}.")

# Transform and append to JSONL file

# # Example usage
input_example = load_examples_from_json(input_json_raw_file)

transform_to_jsonl(input_example, output_jsonl_file)
print(f"Data appended to {output_jsonl_file}.")



