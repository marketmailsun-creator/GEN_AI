import os
from dotenv import load_dotenv
#from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, Docx2txtLoader,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter # For splitting documents into chunks
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap, TavilySearch  
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import asyncio
import ssl
from typing import Any, Dict, List
import certifi
from langchain_core.documents import Document
from langchain_chroma import Chroma


load_dotenv()
#configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
google_api_key = os.getenv("GOOGLE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
index_name = os.getenv("INDEX_NAME_TAV")


#vectorstore = PineconeVectorStore(index_name=index_name, embedding=embedding)
vectorstore = Chroma(persist_directory="chroma_db")
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=100)
tavily_crawl = TavilyCrawl()

async def index_docs_async(docs: List[Document],batch_size: int = 10):
    """Process and index documents in batches asynchronously."""
    print("Indexing documents to Pinecone Vector Store...")

    batches = [
        docs[i : i + batch_size] for i in range(0, len(docs), batch_size)   
    ]
    print("Documents indexed successfully.")

#process batches concurrently
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.aadd_documents(batch)
        except Exception as e:
            print(f"Error indexing batch {batch_num}: {e}")
            return False
        return True
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    successful = sum(1 for r in results if r is True)

    if successful == len(batches):
        print("All batches indexed successfully.")
    else:               
        print(f"Successfully indexed {successful} out of {len(batches)} batches.")
    
async def main():
    res = tavily_crawl.invoke(
        {
            "url": "https://www.interviewbit.com/angular-interview-questions/",
            "max_depth": 1,
            "extract_depth": "advanced", 
        }
    )
    all_docs = [
            Document(
                page_content=result['raw_content'], 
                metadata={"source" : result['url']}
            ) 
            for result in res["results"]
        ]
    
    print(f"Number of documents extracted: {len(all_docs)}")
    doc_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = doc_splitter.split_documents(all_docs) #Multipli options for chunking is available in text_splitter module
    await index_docs_async(split_docs, batch_size=500)

if __name__ == "__main__":
    asyncio.run(main())





    
