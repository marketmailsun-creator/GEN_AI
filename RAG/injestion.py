import os
from dotenv import load_dotenv
#from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, Docx2txtLoader,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import asyncio

load_dotenv()

if __name__ == "__main__":
    # google_api_key = os.getenv("GOOGLE_API_KEY")
    # openai_api_key = os.getenv("OPENAI_API_KEY")
    # index_name = os.getenv("INDEX_NAME")

    # loader = Docx2txtLoader("/home/sam/Desktop/GEN_AI/RAG/suman_resume.docx")
    # documents = loader.load()
    # doc_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # docs = doc_splitter.split_documents(documents)
    # print(f"Number of documents: {len(docs)}")
    # #embedding = OpenAIEmbeddings(openai_api_key = openai_api_key ,model="text-embedding-3-small")
    # embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # print("Creating Pinecone Vector Store...")
    # vectorstore = PineconeVectorStore.from_documents(docs, embedding, index_name=index_name)
    # print("Pinecone Vector Store created successfully.")

    loader_pdf = Docx2txtLoader("/home/sam/Desktop/GEN_AI/RAG/suman_resume.docx")
    documents_doc = loader_pdf.load()
    doc_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_pdf = doc_splitter.split_documents(documents_doc)
    print(f"Number of documents: {len(docs_pdf)}")
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("Creating FAISS Vector Store...")
    vectorstore_faiss = FAISS.from_documents(docs_pdf, embedding)
    vectorstore_faiss.save_local("faiss_index_rag")
    print("FAISS Store created successfully.")




    
