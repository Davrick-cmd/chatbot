####################################################################
#                         import
####################################################################

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import os, glob
from pathlib import Path

# Import openai and google_genai as main LLM services
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# langchain prompts, memory, chains...
from langchain.prompts import PromptTemplate, ChatPromptTemplate

from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory

from langchain.schema import format_document





# OutputParser
from langchain_core.output_parsers import StrOutputParser

# Import chroma as the vector store
from langchain_community.vectorstores import Chroma




# Import streamlit
import streamlit as st

from langchain.schema import format_document,Document
from langchain_core.messages import get_buffer_string
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from operator import itemgetter

from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
# LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
# LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")

firecrawl_api_key = os.getenv("FIREcrawl_API_KEY")


def instantiate_LLM(LLM_provider,api_key,temperature=0.5,top_p=0.95,model_name=None):
    """Instantiate LLM in Langchain.
    Parameters:
        LLM_provider (str): the LLM provider; in ["OpenAI","Google","HuggingFace"]
        model_name (str): in [" gpt-4o", " gpt-4o-0125", "gpt-4-turbo-preview", 
            "gemini-pro", "mistralai/Mistral-7B-Instruct-v0.2"].            
        api_key (str): google_api_key or openai_api_key or huggingfacehub_api_token 
        temperature (float): Range: 0.0 - 1.0; default = 0.5
        top_p (float): : Range: 0.0 - 1.0; default = 1.
    """
    if LLM_provider == "OpenAI":
        llm = ChatOpenAI(
            api_key=api_key,
            model=model_name,
            temperature=temperature,
            top_p=top_p
        )
    return llm


def clean_tmpfiles(TMP_DIR):
    # 1. Delete old temp files
    files = glob.glob(TMP_DIR.as_posix() + "/*")
    for f in files:
        try:
            os.remove(f)
        except:
            pass

    # 2. Upload selected documents to temp directory
    if st.session_state.uploaded_file_list is not None:
        for uploaded_file in st.session_state.uploaded_file_list:
            error_message = ""
            try:
                temp_file_path = os.path.join(
                    TMP_DIR.as_posix(), uploaded_file.name
                )
                with open(temp_file_path, "wb") as temp_file:
                    temp_file.write(uploaded_file.read())
            except Exception as e:
                error_message += e
        if error_message != "":
            st.warning(f"Errors: {error_message}")

def create_memory(model_name=' gpt-4o',memory_max_token=None, base_retriever_fetch_k=None, base_retriever_lambda_mult=None):
    """Creates a ConversationSummaryBufferMemory for  gpt-4o
    Creates a ConversationBufferMemory for the other models."""
    
    if model_name==" gpt-4o":
        if memory_max_token is None:
            memory_max_token = 1024 # max_tokens for ' gpt-4o' = 4096
        memory = ConversationSummaryBufferMemory(
            max_token_limit=memory_max_token,
            llm=ChatOpenAI(model_name=" gpt-4o",openai_api_key=openai_api_key,temperature=0.1),
            return_messages=True,
            memory_key='chat_history',
            output_key="answer",
            input_key="question",
        )
    else:
        memory = ConversationBufferMemory(
            return_messages=True,
            memory_key='chat_history',
            output_key="answer",
            input_key="question",
        )  
    return memory


def answer_template(language="english"):
    """Generate a prompt to pass the standalone question, question type, chat history, and context to the LLM for an accurate and detailed answer."""
    
    template = f"""
You are an expert in answering questions about documents based on the provided context. Your role is to analyze the input and craft a professional, accurate, and contextually relevant response.

Use only the following context (delimited by <context></context>) to formulate your answer. Do not include information that is not found in the context.

<context>
{{chat_history}}
{{context}}
</context>

Question Type: {{question_type}}
Question: {{question}}

Instructions:
- **General Summary**: If the question type is "General Summary," provide a concise yet comprehensive high-level overview of the document content, ensuring all key details are covered.If a user is asking for detailed summary make sure to include more relavant details.
- **Specific Summary**: If the question type is "Specific Summary," focus on summarizing the most relevant section or topic from the document, providing sufficient detail.
- **Specific Question**: If the question type is "Specific Question," deliver a precise and exact answer to the query asked, based strictly on the context provided.
- **General Question**: If the question type is "General Question," offer a broad understanding of the document's content while maintaining relevance and clarity feel free to include external resources.

The response must be professional, detailed, and written in {language}. Avoid speculative or unsupported claims.
"""
    return template

    
def _combine_documents(docs, document_prompt, document_separator="\n\n"):
    print("Number of documents used:",len(docs))
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    
    return document_separator.join(doc_strings)

def custom_ConversationalRetrievalChain(
    llm, condense_question_llm,
    retriever,
    memory,
    language="english",
    llm_provider="OpenAI",
    model_name='gpt-4o',
):
    """Create a ConversationalRetrievalChain step by step.
    """
    ##############################################################
    # Step 1: Create a standalone_question chain
    ##############################################################
    
    # 1. Load memory using RunnableLambda. Retrieves the chat_history attribute using itemgetter.
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history"),
    )

    # 5. Retrieve relevant documents
    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
        "question_type":itemgetter("question_type")
    }
    
    # 6. Get variables ['chat_history', 'context', 'question'] that will be passed to `answer_prompt`
    DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
    answer_prompt = ChatPromptTemplate.from_template(answer_template(language=language)) 
    # 3 variables are expected ['chat_history', 'context', 'question'] by the ChatPromptTemplate   
    answer_prompt_variables = {
        "context": lambda x: _combine_documents(docs=x["docs"], document_prompt=DEFAULT_DOCUMENT_PROMPT),
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history"),
        "question_type":itemgetter("question_type")  # get it from `loaded_memory` variable
    }
    
    # 7. Load memory, format `answer_prompt` with variables (context, question and chat_history) and pass the `answer_prompt to LLM.
    # return answer, docs and standalone_question
    
    chain_answer = {
        "answer": loaded_memory | answer_prompt_variables | answer_prompt | llm,
        # return only page_content and metadata 
        "docs": lambda x: [Document(page_content=doc.page_content, metadata=doc.metadata) for doc in x["docs"]],
        "standalone_question": lambda x: x["question"]  # return standalone_question
    }

    # 8. Final chain including loaded memory
    conversational_retriever_chain = retrieved_documents | loaded_memory | chain_answer

    print("Conversational retriever chain created successfully!")

    return conversational_retriever_chain


def create_standalone_question_and_type(prompt,llm):
    """Create a standalone question and determine its type."""
    ##############################################################
    # Step 1: Create a standalone_question chain
    ##############################################################
    
    # 1. Create memory: ConversationSummaryBufferMemory for gpt-3.5, and ConversationBufferMemory for the other models
    
    # Check if st.session_state.memory exists; if so, use it; otherwise, create a new memory instance
    if hasattr(st.session_state, 'memory') and st.session_state.memory is not None:
        memory = st.session_state.memory
    else:
        memory = create_memory('gpt-4o', base_retriever_fetch_k=None, base_retriever_lambda_mult=None)  # New MMR parameters
    
    # 2. Load memory using RunnableLambda. Retrieves the chat_history attribute using itemgetter.
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history"),
    )

    # 3. Pass the follow-up question along with the chat history to the LLM, and parse the answer (standalone_question).
    condense_question_prompt = PromptTemplate(
        input_variables=['chat_history', 'question'], 
        template="""Given the following conversation and a follow up question, 
    rephrase the follow up question to be a standalone question, in the same language as the follow up question.\n\n
    Chat History:\n{chat_history}\n
    Follow Up Input: {question}\n
    Standalone question:"""        
    )
    
    # New prompt to determine the question type
    question_type_prompt = PromptTemplate(
        input_variables=['standalone_question'],
        template="""You are an expert in categorizing questions people ask about documents. 
    Determine the type of the following question based on the definitions and examples provided:

    Possible types and their definitions:
    1. General_Summary: A high-level overview of the entire content. 
    Example: "Summarize the document."

    2. Specific_Summary: A summary of a specific section or topic within the content.
    Example: "Summarize the section about market trends."

    3. Specific_Question: A direct query requesting a particular piece of information.
    Example: "What is the revenue for Q1?"

    4. General_Question: A broad or open-ended query that requires understanding the content without targeting a specific fact.
    Example: "What does this report talk about?"

    Question: {standalone_question}
    Type of question (only return one of the four types):"""
    )
        
    standalone_question_chain = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | condense_question_prompt
        | llm
        | StrOutputParser(),
    }

    # New chain to determine the question type
    question_type_chain = {
        "standalone_question": itemgetter("standalone_question"),
        "question_type": {
            "standalone_question": itemgetter("standalone_question")
        } | question_type_prompt | llm | StrOutputParser(),
    }
    
    # 4. Combine load_memory and standalone_question_chain
    chain_question = loaded_memory | standalone_question_chain | question_type_chain

    output= chain_question.invoke({"question": prompt})

    print(output)
    return output['standalone_question'],output['question_type']  # Return the output for further use