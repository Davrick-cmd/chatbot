####################################################################
#                         import
####################################################################

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import os, glob
from pathlib import Path



# document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredWordDocumentLoader
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


from dotenv import load_dotenv

load_dotenv()

firecrawl_api_key = os.getenv("FIREcrawl_API_KEY")

from langchain_openai import OpenAIEmbeddings

####################################################################
#        Process documents and create vectorstor (Chroma dB)
####################################################################

TMP_DIR = Path(__file__).resolve().parent.joinpath("docs")
LOCAL_VECTOR_STORE_DIR = (
    Path(__file__).resolve().parent.joinpath("Vector_store")
)

openai_api_key = os.getenv("OPENAI_API_KEY")

# def delte_temp_files():
#     """delete files from the './data/tmp' folder"""
#     files = glob.glob(TMP_DIR.as_posix() + "/*")
#     for f in files:
#         try:
#             os.remove(f)
#         except:
#             pass

def langchain_document_loader():
    """
    Create document loaders for all supported file types in the docs directory.
    Supported types: PDF, TXT, CSV, DOCX
    """
    documents = []

    # Debug: Print the directory being searched
    print(f"Searching for documents in: {TMP_DIR.absolute()}")
    print(f"Directory exists: {TMP_DIR.exists()}")
    print(f"Files in directory: {list(TMP_DIR.glob('*'))}")

    # Load text files
    txt_loader = DirectoryLoader(
        str(TMP_DIR),
        glob="*.txt",
        loader_cls=TextLoader, 
        show_progress=True
    )
    documents.extend(txt_loader.load())

    # Load PDF files
    pdf_loader = DirectoryLoader(
        str(TMP_DIR), 
        glob="*.pdf", 
        loader_cls=PyPDFLoader, 
        show_progress=True
    )
    documents.extend(pdf_loader.load())

    # Load CSV files
    csv_loader = DirectoryLoader(
        str(TMP_DIR), 
        glob="*.csv", 
        loader_cls=CSVLoader, 
        show_progress=True,
        loader_kwargs={"encoding": "utf8"}
    )
    documents.extend(csv_loader.load())

    # Load DOCX files
    doc_loader = DirectoryLoader(
        str(TMP_DIR),
        glob="*.docx",
        loader_cls=UnstructuredWordDocumentLoader,
        show_progress=True,
    )
    documents.extend(doc_loader.load())

    print('Number of Documents:', len(documents))
    return documents



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




def create_vectorstore(
    vector_store_name,
    openai_api_key=None,
):
    """
    Create a vector store from files in the docs directory
    
    Parameters:
        vector_store_name: Name for the vector store
        openai_api_key: OpenAI API key if using OpenAI
    """
    try:
        # 1. Load documents directly from docs directory
        documents = langchain_document_loader()
        print('Documents loaded:', len(documents))

        # 2. Split documents to chunks
        chunks = split_documents_to_chunks(documents)
        print(len(chunks))

        # 3. Create embeddings
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)

        # 4. Create vectorstore
        persist_directory = os.path.join(LOCAL_VECTOR_STORE_DIR.as_posix(), vector_store_name)
        
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory,
        )
        print(f"Vectorstore '{vector_store_name}' created successfully with {len(chunks)} chunks")

        return vector_store

    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    load_dotenv()
    
    # Create the necessary directories if they don't exist
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    
    vector_store = create_vectorstore(
        vector_store_name="BusinessDefinition",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    if vector_store:
        print("Vector store created successfully!")
    else:
        print("Failed to create vector store")


