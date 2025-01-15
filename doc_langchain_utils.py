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

def create_memory(model_name=' gpt-4o',memory_max_token=None):
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
    """Pass the standalone question along with the chat history and context (retrieved documents) to the `LLM` to get an answer."""
    
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
    
def _combine_documents(docs, document_prompt, document_separator="\n\n"):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

def custom_ConversationalRetrievalChain(
    llm,condense_question_llm,
    retriever,
    language="english",
    llm_provider="OpenAI",
    model_name='gpt-4o',
):
    """Create a ConversationalRetrievalChain step by step.
    """
    ##############################################################
    # Step 1: Create a standalone_question chain
    ##############################################################
    
    # 1. Create memory: ConversationSummaryBufferMemory for gpt-3.5, and ConversationBufferMemory for the other models
    
    memory = create_memory(model_name)
    # memory = ConversationBufferMemory(memory_key="chat_history",output_key="answer", input_key="question",return_messages=True)
    # 2. load memory using RunnableLambda. Retrieves the chat_history attribute using itemgetter.
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history"),
    )


    # 3. Pass the follow-up question along with the chat history to the LLM, and parse the answer (standalone_question).

    condense_question_prompt = PromptTemplate(
        input_variables=['chat_history', 'question'], 
        template = """Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question, in the same language as the follow up question.\n\n
Chat History:\n{chat_history}\n
Follow Up Input: {question}\n
Standalone question:"""        
)
        
    standalone_question_chain = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | condense_question_prompt
        | condense_question_llm
        | StrOutputParser(),
    }

    
    # 4. Combine load_memory and standalone_question_chain
    chain_question = loaded_memory | standalone_question_chain

    
    ####################################################################################
    #   Step 2: Retrieve documents, pass them to the LLM, and return the response.
    ####################################################################################

    # 5. Retrieve relevant documents
    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }
    
    # 6. Get variables ['chat_history', 'context', 'question'] that will be passed to `answer_prompt`
    
    DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
    answer_prompt = ChatPromptTemplate.from_template(answer_template(language=language)) 
    # 3 variables are expected ['chat_history', 'context', 'question'] by the ChatPromptTemplate   
    answer_prompt_variables = {
        "context": lambda x: _combine_documents(docs=x["docs"],document_prompt=DEFAULT_DOCUMENT_PROMPT),
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history") # get it from `loaded_memory` variable
    }
    
    # 7. Load memory, format `answer_prompt` with variables (context, question and chat_history) and pass the `answer_prompt to LLM.
    # return answer, docs and standalone_question
    
    chain_answer = {
        "answer": loaded_memory | answer_prompt_variables | answer_prompt | llm,
        # return only page_content and metadata 
        "docs": lambda x: [Document(page_content=doc.page_content,metadata=doc.metadata) for doc in x["docs"]],
        "standalone_question": lambda x:x["question"] # return standalone_question
    }

    # 8. Final chain
    conversational_retriever_chain = chain_question | retrieved_documents | chain_answer

    print("Conversational retriever chain created successfully!")

    return conversational_retriever_chain,memory







