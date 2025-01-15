####################################################################
#                         import
####################################################################

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import os, glob
from pathlib import Path

# Import openai and google_genai as main LLM services
from langchain_openai import OpenAIEmbeddings, ChatOpenAI



from langchain.schema import format_document


# document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    CSVLoader,
    Docx2txtLoader,
    WebBaseLoader,
    FireCrawlLoader
)

# text_splitter
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

# OutputParser
from langchain_core.output_parsers import StrOutputParser

# Import chroma as the vector store
from langchain_community.vectorstores import Chroma

# Contextual_compression
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter,
    LongContextReorder,
)
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever

# Cohere
from langchain.retrievers.document_compressors import CohereRerank


# HuggingFace
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings


# Import streamlit
import streamlit as st

from doc_langchain_utils import instantiate_LLM, custom_ConversationalRetrievalChain
from urllib.parse import urljoin, urlparse
from langchain_community.vectorstores.utils import filter_complex_metadata

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
from langchain.schema import Document

from dotenv import load_dotenv

load_dotenv()

firecrawl_api_key = os.getenv("FIREcrawl_API_KEY")

####################################################################
#        Process documents and create vectorstor (Chroma dB)
####################################################################

TMP_DIR = Path(__file__).resolve().parent.joinpath("data", "tmp")
LOCAL_VECTOR_STORE_DIR = (
    Path(__file__).resolve().parent.joinpath("data", "vector_stores")
)

openai_api_key = os.getenv("OPENAI_API_KEY")

def delte_temp_files():
    """delete files from the './data/tmp' folder"""
    files = glob.glob(TMP_DIR.as_posix() + "/*")
    for f in files:
        try:
            os.remove(f)
        except:
            pass
def sanitize_filename(name):
    """Sanitize the filename to be filesystem safe."""
    return re.sub(r'[<>:"/\\|?*]', '_', name)


    
def get_page_title_from_document(document):
    """Extract a suitable page name from the document metadata or URL."""
    metadata = document.metadata.get("source", "")
    title = document.metadata.get("title", "")  # Assuming metadata contains title (if available)
    if title:
        return sanitize_filename(title)
    return sanitize_filename(metadata.split("/")[-1] or "index")
    
def save_content_to_file(content, filename, output_dir="scraped_pages"):
    """Save content to a text file."""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename + '.txt')
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def clean_page_content(soup):
    """Remove styles, scripts, and other non-content elements from the soup."""
    for script_or_style in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
        script_or_style.decompose()  # Remove these tags and their content
    text = soup.get_text(separator=' ')  # Extract text content
    lines = [line.strip() for line in text.splitlines()]  # Remove extra whitespace
    text = ' '.join([line for line in lines if line])  # Rejoin lines with non-empty content
    return text

def filter_metadata(metadata, allowed_keys=None):
    """Filter metadata to include only allowed keys."""
    allowed_keys = allowed_keys or {'description', 'title', 'language'}  # Keys to keep
    if isinstance(metadata, dict):
        return {key: value for key, value in metadata.items() if key in allowed_keys}
    else:
        return metadata

def langchain_document_loader():
    """
    Crete documnet loaders for PDF, TXT and CSV files.
    https://python.langchain.com/docs/modules/data_connection/document_loaders/file_directory
    """
    documents = []
    if st.session_state.uploaded_file_list and not st.session_state.website:
        

        txt_loader = DirectoryLoader(
            TMP_DIR.as_posix(), glob="**/*.txt", loader_cls=TextLoader, show_progress=True
        )
        documents.extend(txt_loader.load())

        pdf_loader = DirectoryLoader(
            TMP_DIR.as_posix(), glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True
        )
        documents.extend(pdf_loader.load())

        csv_loader = DirectoryLoader(
            TMP_DIR.as_posix(), glob="**/*.csv", loader_cls=CSVLoader, show_progress=True,
            loader_kwargs={"encoding":"utf8"}
        )
        documents.extend(csv_loader.load())

        doc_loader = DirectoryLoader(
            TMP_DIR.as_posix(),
            glob="**/*.docx",
            loader_cls=Docx2txtLoader,
            show_progress=True,
        )
        documents.extend(doc_loader.load())
    elif st.session_state.website and not st.session_state.uploaded_file_list:
        
        # # Load website content (add the URL of the website you want to scrape)
        # website_loader = FireCrawlLoader(
        #     api_key=firecrawl_api_key,url = st.session_state.website,mode='crawl'
        # )


        # Example of loading and processing documents
        website_loader = FireCrawlLoader(
            api_key=firecrawl_api_key, url=st.session_state.website, mode='crawl'
        )
        # documents = website_loader.load()

        # # Filter and flatten metadata if necessary
        # processed_documents = []
        # data = website_loader.load()
        # documents.extend(data)
        # print('Trying metadata:',[data[0].metadata][0])
        # data[0].metadata = filter_metadata([data[0].metadata][0])
        # print('Filtered metadata:', data[0].metadata)
        # # test = filter_complex_metadata(data[0])
        # # print(test)

        for i,doc in enumerate(website_loader.load()):
            if isinstance(doc, Document):
                # Flatten and filter metadata if complex
                print([doc.metadata][0])
                doc.metadata = filter_metadata([doc.metadata][0])
                documents.append(doc)
                print('Done with document' + str(i))
            else:
                # Handle or skip unexpected types if not a Document
                print(doc)
                print(f"Skipping non-Document object: {type(doc)}")


        # def get_all_page_urls(base_url):
        #     """Crawl a website and return a list of URLs."""
        #     visited = set()
        #     to_visit = [base_url]
        #     all_urls = []

        #     while to_visit:
        #         url = to_visit.pop()
        #         if url not in visited:
        #             visited.add(url)
        #             all_urls.append(url)
        #             try:
        #                 response = requests.get(url)
        #                 if response.status_code == 200:
        #                     soup = BeautifulSoup(response.text, 'html.parser')
        #                     links = soup.find_all('a', href=True)
        #                     for link in links:
        #                         full_url = urljoin(url, link['href'])
        #                         if base_url in full_url and full_url not in visited:
        #                             to_visit.append(full_url)
        #             except Exception as e:
        #                 print(f"Failed to fetch {url}: {e}")
        #     print('Number of Url visited are:',len(all_urls))
        #     return all_urls

        # # Example usage
        # base_url = st.session_state.website
        # urls_to_scrape = get_all_page_urls(base_url)
        # def scrape_website(urls):
        #     documents = []
        #     for url in urls:
        #         try:
        #             loader = FireCrawlLoader(api_key=firecrawl_api_key,url=url,mode="crawl")
        #             data = loader.load()
        #             documents.extend(data)
        #         except Exception as e:
        #             print(f"Failed to scrape {url}: {e}")
        # # Scrape content from all URLs
        # scrape_website(urls_to_scrape)
    else:
        st.warning('Select the source') 
    print('Number of Documents ',len(documents))
    return documents

# def langchain_document_loader():
#     """
#     Create document loaders for PDF, TXT, CSV files, and a website.
#     https://python.langchain.com/docs/modules/data_connection/document_loaders/file_directory
#     """

#     documents = []

#     # Load TXT files
#     txt_loader = DirectoryLoader(
#         TMP_DIR.as_posix(), glob="**/*.txt", loader_cls=TextLoader, show_progress=True
#     )
#     documents.extend(txt_loader.load())

#     # Load PDF files
#     pdf_loader = DirectoryLoader(
#         TMP_DIR.as_posix(), glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True
#     )
#     documents.extend(pdf_loader.load())

#     # Load CSV files
#     csv_loader = DirectoryLoader(
#         TMP_DIR.as_posix(), glob="**/*.csv", loader_cls=CSVLoader, show_progress=True,
#         loader_kwargs={"encoding":"utf8"}
#     )
#     documents.extend(csv_loader.load())

#     # Load DOCX files
#     doc_loader = DirectoryLoader(
#         TMP_DIR.as_posix(),
#         glob="**/*.docx",
#         loader_cls=Docx2txtLoader,
#         show_progress=True,
#     )
#     documents.extend(doc_loader.load())

#     # Load website content (add the URL of the website you want to scrape)
#     website_loader = WebBaseLoader(
#         "https://bk.rw/", show_progress=True
#     )
#     documents.extend(website_loader.load())

#     def get_all_page_urls(base_url):
#         """Crawl a website and return a list of URLs."""
#         visited = set()
#         to_visit = [base_url]
#         all_urls = []

#         while to_visit:
#             url = to_visit.pop()
#             if url not in visited:
#                 visited.add(url)
#                 all_urls.append(url)
#                 try:
#                     response = requests.get(url)
#                     if response.status_code == 200:
#                         soup = BeautifulSoup(response.text, 'html.parser')
#                         links = soup.find_all('a', href=True)
#                         for link in links:
#                             full_url = urljoin(url, link['href'])
#                             if base_url in full_url and full_url not in visited:
#                                 to_visit.append(full_url)
#                 except Exception as e:
#                     print(f"Failed to fetch {url}: {e}")
#         return all_urls

#     # Example usage
#     base_url = "https://bk.rw"
#     urls_to_scrape = get_all_page_urls(base_url)
#     def scrape_website(urls):
#         documents = []
#         for url in urls:
#             try:
#                 loader = WebBaseLoader(url)
#                 documents.extend(loader.load())
#             except Exception as e:
#                 print(f"Failed to scrape {url}: {e}")
#         return documents

#     # Scrape content from all URLs
#     scrape_website(urls_to_scrape)
#     return documents



def split_documents_to_chunks(documents):
    """Split documents to chunks using RecursiveCharacterTextSplitter."""

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    print('Number of Chunks',len(chunks))
    return chunks


def select_embeddings_model():
    """Select embeddings models: OpenAIEmbeddings or GoogleGenerativeAIEmbeddings."""
    if st.session_state.LLM_provider == "OpenAI":
        embeddings = OpenAIEmbeddings(api_key=st.session_state.openai_api_key)

    if st.session_state.LLM_provider == "HuggingFace":
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=st.session_state.hf_api_key, model_name="thenlper/gte-large"
        )

    return embeddings


def create_retriever(
    vector_store,
    embeddings,
    retriever_type="Contextual compression",
    base_retriever_search_type="semilarity",
    base_retriever_k=16,
    compression_retriever_k=20,
    cohere_api_key="",
    cohere_model="rerank-multilingual-v2.0",
    cohere_top_n=10,
):
    """
    create a retriever which can be a:
        - Vectorstore backed retriever: this is the base retriever.
        - Contextual compression retriever: We wrap the the base retriever in a ContextualCompressionRetriever.
            The compressor here is a Document Compressor Pipeline, which splits documents
            to smaller chunks, removes redundant documents, filters the top relevant documents,
            and reorder the documents so that the most relevant are at beginning / end of the list.
        - Cohere_reranker: CohereRerank endpoint is used to reorder the results based on relevance.

    Parameters:
        vector_store: Chroma vector database.
        embeddings: OpenAIEmbeddings or GoogleGenerativeAIEmbeddings.

        retriever_type (str): in [Vectorstore backed retriever,Contextual compression,Cohere reranker]. default = Cohere reranker

        base_retreiver_search_type: search_type in ["similarity", "mmr", "similarity_score_threshold"], default = similarity.
        base_retreiver_k: The most similar vectors are returned (default k = 16).

        compression_retriever_k: top k documents returned by the compression retriever, default = 20

        cohere_api_key: Cohere API key
        cohere_model (str): model used by Cohere, in ["rerank-multilingual-v2.0","rerank-english-v2.0"]
        cohere_top_n: top n documents returned bu Cohere, default = 10

    """

    base_retriever = Vectorstore_backed_retriever(
        vectorstore=vector_store,
        search_type=base_retriever_search_type,
        k=base_retriever_k,
        score_threshold=None,
    )

    if retriever_type == "Vectorstore backed retriever":
        return base_retriever

    elif retriever_type == "Contextual compression":
        compression_retriever = create_compression_retriever(
            embeddings=embeddings,
            base_retriever=base_retriever,
            k=compression_retriever_k,
        )
        return compression_retriever

    elif retriever_type == "Cohere reranker":
        cohere_retriever = CohereRerank_retriever(
            base_retriever=base_retriever,
            cohere_api_key=cohere_api_key,
            cohere_model=cohere_model,
            top_n=cohere_top_n,
        )
        return cohere_retriever
    else:
        pass


def Vectorstore_backed_retriever(
    vectorstore, search_type="similarity", k=4, score_threshold=None
):
    """create a vectorsore-backed retriever
    Parameters:
        search_type: Defines the type of search that the Retriever should perform.
            Can be "similarity" (default), "mmr", or "similarity_score_threshold"
        k: number of documents to return (Default: 4)
        score_threshold: Minimum relevance threshold for similarity_score_threshold (default=None)
    """
    search_kwargs = {}
    if k is not None:
        search_kwargs["k"] = k
    if score_threshold is not None:
        search_kwargs["score_threshold"] = score_threshold

    retriever = vectorstore.as_retriever(
        search_type=search_type, search_kwargs=search_kwargs
    )
    return retriever


def create_compression_retriever(
    embeddings, base_retriever, chunk_size=500, k=16, similarity_threshold=None
):
    """Build a ContextualCompressionRetriever.
    We wrap the the base_retriever (a Vectorstore-backed retriever) in a ContextualCompressionRetriever.
    The compressor here is a Document Compressor Pipeline, which splits documents
    to smaller chunks, removes redundant documents, filters the top relevant documents,
    and reorder the documents so that the most relevant are at beginning / end of the list.

    Parameters:
        embeddings: OpenAIEmbeddings or GoogleGenerativeAIEmbeddings.
        base_retriever: a Vectorstore-backed retriever.
        chunk_size (int): Docs will be splitted into smaller chunks using a CharacterTextSplitter with a default chunk_size of 500.
        k (int): top k relevant documents to the query are filtered using the EmbeddingsFilter. default =16.
        similarity_threshold : similarity_threshold of the  EmbeddingsFilter. default =None
    """

    # 1. splitting docs into smaller chunks
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=0, separator=". "
    )

    # 2. removing redundant documents
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)

    # 3. filtering based on relevance to the query
    relevant_filter = EmbeddingsFilter(
        embeddings=embeddings, k=k, similarity_threshold=similarity_threshold
    )

    # 4. Reorder the documents

    # Less relevant document will be at the middle of the list and more relevant elements at beginning / end.
    # Reference: https://python.langchain.com/docs/modules/data_connection/retrievers/long_context_reorder
    reordering = LongContextReorder()

    # 5. create compressor pipeline and retriever
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter, reordering]
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=base_retriever
    )

    return compression_retriever


def CohereRerank_retriever(
    base_retriever, cohere_api_key, cohere_model="rerank-multilingual-v2.0", top_n=10
):
    """Build a ContextualCompressionRetriever using CohereRerank endpoint to reorder the results
    based on relevance to the query.

    Parameters:
       base_retriever: a Vectorstore-backed retriever
       cohere_api_key: the Cohere API key
       cohere_model: the Cohere model, in ["rerank-multilingual-v2.0","rerank-english-v2.0"], default = "rerank-multilingual-v2.0"
       top_n: top n results returned by Cohere rerank. default = 10.
    """

    compressor = CohereRerank(
        cohere_api_key=cohere_api_key, model=cohere_model, top_n=top_n
    )

    retriever_Cohere = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )
    return retriever_Cohere


def retrieval_blocks(
    LLM_service="OpenAI",
    vectorstore_name="Vit_All_OpenAI_Embeddings",
    embeddings='Vectorstore backed retriever',  
    retriever_type="Vectorstore_backed_retriever",
    base_retriever_search_type="similarity", base_retriever_k=10, base_retriever_score_threshold=None,
    compression_retriever_k=16,
    cohere_api_key="***", cohere_model="rerank-multilingual-v2.0", cohere_top_n=8,
):
    """
    Rertieval includes: document loaders, text splitter, vectorstore and retriever. 
    
    Parameters: 
        create_vectorstore (boolean): If True, a new Chroma vectorstore will be created. Otherwise, an existing vectorstore will be loaded.
        LLM_service: OpenAI, Google or HuggingFace.
        vectorstore_name (str): the name of the vectorstore.
        chunk_size and chunk_overlap: parameters of the RecursiveCharacterTextSplitter, default = (1600,200).
        
        retriever_type (str): in [Vectorstore_backed_retriever,Contextual_compression,Cohere_reranker]
        
        base_retriever_search_type: search_type in ["similarity", "mmr", "similarity_score_threshold"], default = similarity.
        base_retriever_k: The most similar vectors to retrieve (default k = 10).  
        base_retriever_score_threshold: score_threshold used by the base retriever, default = None.

        compression_retriever_k: top k documents returned by the compression retriever, default=16
        
        cohere_api_key: Cohere API key
        cohere_model (str): The Cohere model can be either 'rerank-english-v2.0' or 'rerank-multilingual-v2.0', with the latter being the default.
        cohere_top_n: top n results returned by Cohere rerank, default = 8.
   
    Output:
        retriever.
    """
    try:

        embeddings = select_embeddings_model()        
        vector_store = Chroma(
            persist_directory = LOCAL_VECTOR_STORE_DIR.as_posix() + "/" + vectorstore_name,
            embedding_function=embeddings
        )
            
        # 6. base retriever: Vector store-backed retriever 
        base_retriever = Vectorstore_backed_retriever(
            vector_store,
            search_type=base_retriever_search_type,
            k=base_retriever_k,
            score_threshold=base_retriever_score_threshold
        )
        retriever = None
        if retriever_type=="Vectorstore backed retriever": 
            retriever = base_retriever
    
        # 7. Contextual Compression Retriever
        if retriever_type=="Contextual compression":    
            retriever = create_compression_retriever(
                embeddings=embeddings,
                base_retriever=base_retriever,
                k=compression_retriever_k,
            )
    
        # 8. CohereRerank retriever
        if retriever_type=="Cohere reranker":
            retriever = CohereRerank_retriever(
                base_retriever=base_retriever, 
                cohere_api_key=cohere_api_key, 
                cohere_model=cohere_model, 
                top_n=cohere_top_n
            )
 
        st.success(f"\n{retriever_type} is created successfully!")
        st.success(f"Relevant documents will be retrieved from vectorstore ({vectorstore_name}) which uses {LLM_service} embeddings \
and has {vector_store._collection.count()} chunks.")
        
        return retriever
    except Exception as e:
        print(e)




def create_vectorstore():
    print('About to create a vectorstore')

    with st.spinner("Creating vectorstore..."):
        # Check inputs
        error_messages = []
        print('Error message',error_messages)
        if (
            not st.session_state.openai_api_key
        ):
            error_messages.append(
                f"insert your {st.session_state.LLM_provider} API key"
            )
            error_messages.append(f"insert your Cohere API key")
        if not st.session_state.uploaded_file_list and not st.session_state.website:
            print('NOT EXISTING',st.session_state.website)
            error_messages.append("select documents to upload or site to search")
        if st.session_state.vector_store_name == "":
            error_messages.append("provide a Vectorstore name")

        if len(error_messages) == 1:
            st.session_state.error_message = "Please " + error_messages[0] + "."
        elif len(error_messages) > 1:
            st.session_state.error_message = (
                "Please "
                + ", ".join(error_messages[:-1])
                + ", and "
                + error_messages[-1]
                + "."
            )
        else:
            try:
                # 1. Delete old temp files
                delte_temp_files()
                # Create new Vectorstore (Chroma index)

                # 2. Upload selected documents to temp directory
                print('uploaded files:',st.session_state.uploaded_file_list)
                if st.session_state.uploaded_file_list is not None and len(st.session_state.uploaded_file_list)>0:
                    print('Begining loop')
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
                print('About to start loading documents',st.session_state.website)
 
               # 4. Load documents with Langchain loaders
                documents = langchain_document_loader()
                print('Documents are:',len(documents))

                # 5. Split documents to chunks
                chunks = split_documents_to_chunks(documents)
                # 6. Embeddings
                embeddings = select_embeddings_model()

                # 7. Create a vectorstore
                persist_directory = (
                    LOCAL_VECTOR_STORE_DIR.as_posix()
                    + "/"
                    + st.session_state.vector_store_name
                )

                try:
                    st.session_state.vector_store = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        persist_directory=persist_directory,
                    )
                    st.info(
                        f"Vectorstore **{st.session_state.vector_store_name}** is created successfully."
                    )

                    # 7. Create retriever
                    st.session_state.retriever = create_retriever(
                        vector_store=st.session_state.vector_store,
                        embeddings=embeddings,
                        retriever_type=st.session_state.retriever_type,
                        base_retriever_search_type="similarity",
                        base_retriever_k=16,
                        compression_retriever_k=20,
                        cohere_api_key=st.session_state.cohere_api_key,
                        cohere_model="rerank-multilingual-v2.0",
                        cohere_top_n=10,
                    )

                    # 8. Create memory and ConversationalRetrievalChain
                    (
                        st.session_state.chain,
                        st.session_state.memory,
                    ) = custom_ConversationalRetrievalChain(
                        llm = instantiate_LLM(
                            LLM_provider="OpenAI",model_name="gpt-4o",api_key=openai_api_key,temperature=0.5
                        ),
                        condense_question_llm = instantiate_LLM(
                            LLM_provider="OpenAI",model_name="gpt-4o",api_key=openai_api_key,temperature=0.1
                        ), 
                        retriever=st.session_state.retriever,
                        language=st.session_state.assistant_language,
                        llm_provider="OpenAI",
                        model_name="gpt-4o"
                    )
                    
                    # 9. Cclear chat_history
                    clear_chat_history()

                except Exception as e:
                    st.error(e)


                            
                # return vectorstore_name
            except Exception as e:
                print(e)

def clear_chat_history():
    """clear chat history and memory."""
    # 1. re-initialize messages
    st.session_state.messages = [
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

dict_welcome_message = {
    "english": "How can I assist you today?",
    "french": "Comment puis-je vous aider aujourd’hui ?",
    "spanish": "¿Cómo puedo ayudarle hoy?",
    "german": "Wie kann ich Ihnen heute helfen?",
    "russian": "Чем я могу помочь вам сегодня?",
    "chinese": "我今天能帮你什么？",
    "arabic": "كيف يمكنني مساعدتك اليوم؟",
    "portuguese": "Como posso ajudá-lo hoje?",
    "italian": "Come posso assistervi oggi?",
    "Japanese": "今日はどのようなご用件でしょうか?",
}