####################################################################
#                         import
####################################################################

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import os, glob
from pathlib import Path

# Import openai and google_genai as main LLM services
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Import streamlit
import streamlit as st

from vector import create_vectorstore,retrieval_blocks
from doc_langchain_utils import instantiate_LLM,custom_ConversationalRetrievalChain,create_memory,create_standalone_question_and_type

from dotenv import load_dotenv

load_dotenv()




dict_welcome_message = {
    "english": "How can I assist you today?",
    "french": "Comment puis-je vous aider aujourd'hui ?",
    "spanish": "¿Cómo puedo ayudarle hoy?",
    "german": "Wie kann ich Ihnen heute helfen?",
    "russian": "Чем я могу помочь вам сегодня?",
    "chinese": "我今天能帮你什么？",
    "arabic": "كيف يمكنني مساعدتك اليوم؟",
    "portuguese": "Como posso ajudá-lo hoje?",
    "italian": "Come posso assistervi oggi?",
    "Japanese": "今日はどのようなご用件でしょうか?",
}



list_retriever_types = [
    "Cohere reranker",
    "Contextual compression",
    "Vectorstore backed retriever",
]

ASSETS_DIR = Path("img")
USER_AVATAR = ASSETS_DIR / "user-icon.png"
BOT_AVATAR = ASSETS_DIR / "chatgpt-icon.png"

TMP_DIR = Path(__file__).resolve().parent.joinpath("data", "tmp")
LOCAL_VECTOR_STORE_DIR = (
    Path(__file__).resolve().parent.joinpath("data", "vector_stores")
)

openai_api_key = os.getenv("OPENAI_API_KEY")
# LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
# LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")
embeddings = OpenAIEmbeddings(api_key=openai_api_key)
st.session_state.cohere_api_key = cohere_api_key


####################################################################
#            Create app interface with streamlit
####################################################################
# st.set_page_config(page_title="Chat With Your Data")

# st.title("BK DocAI")

# API keys
st.session_state.openai_api_key = openai_api_key
st.session_state.cohere_api_key = cohere_api_key

def expander_model_parameters(
    LLM_provider="OpenAI",
    text_input_API_key="OpenAI API Key - [Get an API key](https://platform.openai.com/account/api-keys)",
    list_models=["gpt-4o-0125", "gpt-4o", "gpt-4-turbo-preview"],
):
    """Add a text_input (for API key) and a streamlit expander containing models and parameters."""
    st.session_state.LLM_provider = LLM_provider
    st.session_state.selected_model = "gpt-4o"
    # LLM_provider = "OpenAI"
    # st.session_state.openai_api_key = openai_api_key
    # with st.expander("**Models and parameters**"):
    #     st.session_state.selected_model = st.selectbox(
    #         f"Choose {LLM_provider} model", list_models
    #     )

    #     # model parameters
    #     st.session_state.temperature = st.slider(
    #         "temperature",
    #         min_value=0.0,
    #         max_value=1.0,
    #         value=0.5,
    #         step=0.1,
    #     )
    #     st.session_state.top_p = st.slider(
    #         "top_p",
    #         min_value=0.0,
    #         max_value=1.0,
    #         value=0.95,
    #         step=0.05,
    #     )



def sidebar_and_documentChooser():
    """Create the sidebar and the a tabbed pane: the first tab contains a document chooser (create a new vectorstore);
    the second contains a vectorstore chooser (open an old vectorstore)."""

    with st.sidebar:
        # st.caption("🚀 Welcome to the Bank of Kigali Ask Doc AI System!,an intelligent system to answer questions related to your documents")

        st.write("")
        st.button("Clear Chat History", on_click=clear_chat_history)



        st.divider()
        expander_model_parameters(
            LLM_provider="OpenAI",
            text_input_API_key="OpenAI API Key - [Get an API key](https://platform.openai.com/account/api-keys)",
            list_models=[
                " gpt-4o-0125",
                " gpt-4o",
                "gpt-4-turbo-preview",
            ],
        )
        # Assistant language
        st.write("")
        st.session_state.assistant_language = st.selectbox(
            f"Assistant language", list(dict_welcome_message.keys())
        )

        st.divider()
        st.subheader("Retrievers")
        retrievers = list_retriever_types
        if st.session_state.selected_model == "gpt-4o":
            # for " gpt-4o", we will not use the vectorstore backed retriever
            # there is a high risk of exceeding the max tokens limit (4096).
            retrievers = list_retriever_types[:-1]

        st.session_state.retriever_type = st.selectbox(
            f"Select retriever type", retrievers
        )
        st.write("")
        st.write("\n\n")

    # Tabbed Pane: Create a new Vectorstore | Open a saved Vectorstore
    # Create tabs based on user role
    if  st.session_state.get("is_admin", False):  # Default to False if not set
        tab_open_vectorstore,tab_new_vectorstore = st.tabs(["Select Document Store","Create Document Store" ])
    else:
        tab_open_vectorstore = st.tabs(["Select Vectorstore"])[0]

    # tab_open_vectorstore,tab_new_vectorstore, = st.tabs(
    #     ["Open a saved Vectorstore","Create a new Vectorstore"]
    # )
    with tab_open_vectorstore:
        # Get list of available vectorstores from the directory
        vector_stores_path = Path(__file__).resolve().parent.joinpath("data", "vector_stores")
        available_vectorstores = [d.name for d in vector_stores_path.iterdir() if d.is_dir()]

        if not available_vectorstores:
            st.info("No Document Store found. Please create a new vectorstore first.")
        else:
            st.write("**Select Document Store:**")
            selected_store = st.selectbox(
                label="Available Document Stores",
                options=available_vectorstores,
                placeholder="Select Document Store...",
                label_visibility="collapsed",
                index=None,
            )

            if st.button("Load Selected Document Store"):
                with st.spinner("Loading Document Store..."):
                    st.session_state.selected_vectorstore_name = selected_store
                    try:
                        # Create memory once
                        memory = create_memory('gpt-4o')
                        st.session_state.memory = memory

                        # Create retriever chains
                        st.session_state.retriever_general_summary = retrieval_blocks(
                            vectorstore_name=st.session_state.selected_vectorstore_name,
                            embeddings=embeddings,
                            retriever_type=st.session_state.retriever_type,
                            base_retriever_search_type="mmr",
                            base_retriever_k=80,  # Default value, will be overridden by _get_retrieval_params
                            base_retriever_score_threshold=None,  # Default value, will be overridden
                            base_retriever_fetch_k=None,  # New MMR parameter
                            base_retriever_lambda_mult=None,  # New MMR parameter
                            compression_retriever_k=80,
                            cohere_api_key=st.session_state.cohere_api_key,
                            cohere_model="rerank-multilingual-v2.0",
                            cohere_top_n=80
                        )
                        
                        st.session_state.retriever_question_specific = retrieval_blocks(
                            vectorstore_name=st.session_state.selected_vectorstore_name,
                            embeddings=embeddings,
                            retriever_type=st.session_state.retriever_type,
                            base_retriever_search_type="similarity",
                            base_retriever_k=10,  # Default value, will be overridden by _get_retrieval_params
                            base_retriever_score_threshold=None,  # Default value, will be overridden
                            compression_retriever_k=5,
                            cohere_api_key=st.session_state.cohere_api_key,
                            cohere_model="rerank-multilingual-v2.0",
                            cohere_top_n=5,
                        )

                        st.session_state.retriever_summary_specific = retrieval_blocks(
                            vectorstore_name=st.session_state.selected_vectorstore_name,
                            embeddings=embeddings,
                            retriever_type=st.session_state.retriever_type,
                            base_retriever_search_type="similarity",
                            base_retriever_k=30,  # Default value, will be overridden by _get_retrieval_params
                            base_retriever_score_threshold=None,  # Default value, will be overridden
                            compression_retriever_k=30,
                            cohere_api_key=st.session_state.cohere_api_key,
                            cohere_model="rerank-multilingual-v2.0",
                            cohere_top_n=30,
                        )
                        
                        # Create three different chains sharing the same memory
                        st.session_state.chain_general_summary = custom_ConversationalRetrievalChain(
                            llm=instantiate_LLM(
                                LLM_provider="OpenAI",
                                model_name="gpt-4o",
                                api_key=openai_api_key,
                                temperature=0.1
                            ),
                            condense_question_llm=instantiate_LLM(
                                LLM_provider="OpenAI",
                                model_name="gpt-4o",
                                api_key=openai_api_key,
                                temperature=0.1
                            ),
                            retriever=st.session_state.retriever_general_summary,
                            language=st.session_state.assistant_language,
                            llm_provider="OpenAI",
                            model_name="gpt-4o",
                            memory=memory  # Share memory
                        )

                        st.session_state.chain_question_specific = custom_ConversationalRetrievalChain(
                            llm=instantiate_LLM(
                                LLM_provider="OpenAI",
                                model_name="gpt-4o",
                                api_key=openai_api_key,
                                temperature=0.1
                            ),
                            condense_question_llm=instantiate_LLM(
                                LLM_provider="OpenAI",
                                model_name="gpt-4o",
                                api_key=openai_api_key,
                                temperature=0.1
                            ),
                            retriever=st.session_state.retriever_question_specific,
                            language=st.session_state.assistant_language,
                            llm_provider="OpenAI",
                            model_name="gpt-4o",
                            memory=memory  # Share memory
                        )

                        st.session_state.chain_summary_specific = custom_ConversationalRetrievalChain(
                            llm=instantiate_LLM(
                                LLM_provider="OpenAI",
                                model_name="gpt-4o",
                                api_key=openai_api_key,
                                temperature=0.1
                            ),
                            condense_question_llm=instantiate_LLM(
                                LLM_provider="OpenAI",
                                model_name="gpt-4o",
                                api_key=openai_api_key,
                                temperature=0.1
                            ),
                            retriever=st.session_state.retriever_summary_specific,
                            language=st.session_state.assistant_language,
                            llm_provider="OpenAI",
                            model_name="gpt-4o",
                            memory=memory  # Share memory
                        )

                        clear_chat_history()
                        st.success(f"Successfully loaded Document Store: {selected_store}")
                    except Exception as e:
                        st.error(f"Error loading Document Store: {str(e)}")


    if  st.session_state.get("is_admin", False):

        with tab_new_vectorstore:

            st.session_state.website = st.text_input(
                label='search a site',
                placeholder="Website Link"
            )

            # 1. Select documnets
            st.session_state.uploaded_file_list = st.file_uploader(
                label="**Select documents**",
                accept_multiple_files=True,
                type=(["pdf", "txt", "docx", "csv"]),
            )
            # 2. Process documents
            st.session_state.vector_store_name = st.text_input(
                label="**Documents will be loaded, embedded and ingested into a vectorstore (Chroma dB). Please provide a valid dB name.**",
                placeholder="Documents name",
            )
            # 3. Add a button to process documnets and create a Chroma vectorstore

            st.button("Create Document Store", on_click=create_vectorstore)
            try:
                if st.session_state.error_message != "":
                    st.warning(st.session_state.error_message)
            except:
                pass


def clear_chat_history():
    """clear chat history and memory."""
    # 1. re-initialize messages
    st.session_state.docmessages = [
        {
            "role": "assistant",
            "content": dict_welcome_message[st.session_state.assistant_language],
        }
    ]
    # 2. Clear memory (history)
    try:
        st.session_state.memory.clear()
    except:
        pass

def get_response_from_LLM(prompt):
    """Invoke the LLM, get response, and display results (answer and source documents)."""
    try:
        # 1. Create standalone question and determine its type
        standalone_question, question_type = create_standalone_question_and_type(prompt,llm=instantiate_LLM(
                                LLM_provider="OpenAI",
                                model_name="gpt-4o",
                                api_key=openai_api_key,
                                temperature=0.1
                            ))

        # 2. Select the appropriate retrieval chain based on the question type
        if question_type == "General_Summary":
            retrieval_chain = st.session_state.chain_general_summary
        elif question_type == "Specific_Summary":
            retrieval_chain = st.session_state.chain_summary_specific
        elif question_type == "Specific_Question":
            retrieval_chain = st.session_state.chain_question_specific
        else:
            retrieval_chain = st.session_state.chain_question_specific

        # 3. Invoke the selected retrieval chain
        response = retrieval_chain.invoke({'standalone_question': standalone_question,"question_type":question_type})

        answer = response["answer"].content
        st.session_state.memory.save_context({"question": prompt}, {"answer": answer})  # update memory
        
        # 4. Display results
        st.session_state.docmessages.append({"role": "user", "content": prompt})
        st.session_state.docmessages.append({"role": "assistant", "content": answer})
        
        # st.chat_message("user", avatar=str(USER_AVATAR)).write(prompt)
        with st.chat_message("assistant", avatar=str(BOT_AVATAR)):
            # 4.1. Display answer:
            st.markdown(answer)

            # 4.2. Display source documents:
            with st.expander("**Source documents**"):
                documents_content = ""
                for document in response['docs']:
                    try:
                        page = " (Page: " + str(document.metadata["page"]) + ")"
                    except:
                        page = ""
                    documents_content += (
                        "**Source: "
                        + page
                        + "**\n\n"
                    )
                    documents_content += (
                        "Page content: " + document.page_content + "\n\n\n"
                    )

                st.markdown(documents_content)

    except Exception as e:
        st.warning(e)

def chatbot():
    sidebar_and_documentChooser()
    st.divider()

# Add header showing current vectorstore
    if st.session_state.get("selected_vectorstore_name"):
        st.markdown(
            f"""
            <div style="
                padding: 1rem; 
                border-radius: 0.5rem; 
                background-color: #002060;  /* Bank of Kigali blue */
                color: white; 
                margin: 1rem 0; 
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                <h3 style="margin: 0; font-size: 1.5rem;">
                    🤖 Chatting with: {st.session_state.selected_vectorstore_name}
                </h3>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="
                padding: 1rem; 
                border-radius: 0.5rem; 
                background-color: #f7f7f7; 
                border: 2px solid #002060; 
                color: #002060;
                text-align: center;">
                <strong>Please select a Document Store to start chatting</strong>
            </div>
            """,
            unsafe_allow_html=True
        )



    # Display chat messages
    if "docmessages" not in st.session_state or st.session_state.docmessages is None:
        st.session_state["docmessages"] = [
            {
                "role": "assistant",
                "content": dict_welcome_message[st.session_state.assistant_language],
            }
        ]

        # Disable chat input if no vectorstore is selected
    if st.session_state.get("selected_vectorstore_name") or st.session_state.get("selected_vectorstore_name") is not None:    
        for msg in st.session_state.docmessages:
            if msg["role"] == "user":
                st.chat_message(msg["role"], avatar=str(USER_AVATAR)).write(msg["content"])
            else:
                st.chat_message(msg["role"], avatar=str(BOT_AVATAR)).write(msg["content"])

    # Disable chat input if no vectorstore is selected
    if not st.session_state.get("selected_vectorstore_name") or st.session_state.get("selected_vectorstore_name") is None:
        st.chat_input(placeholder="Please select a Document Store first...", disabled=True)
    elif prompt := st.chat_input(f"Hi {st.session_state.get('firstname', 'there')}, "
        f"what questions can I answer today related to?"):
        if (
            not st.session_state.openai_api_key
        ):
            st.info(
                f"Please insert your {st.session_state.LLM_provider} API key to continue."
            )
            st.stop()
        st.chat_message("user", avatar=str(USER_AVATAR)).write(prompt)
        with st.spinner("Generating response..."):

            get_response_from_LLM(prompt=prompt)

if __name__ == "__main__":
    chatbot()
