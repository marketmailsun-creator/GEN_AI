import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langsmith import Client
from typing import Any, Dict, List


load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
index_name = os.getenv("INDEX_NAME_TAV")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")   

def run_llm(query: str,chat_history: List[Dict[str, Any]] = []):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding
    )
    llm = ChatOpenAI(model="gpt-4.1-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7)
    client = Client(api_key=LANGSMITH_API_KEY)
    #prompt = client.pull_prompt("langchain-ai/retrieval-qa-chat", include_model=True)
    prompt = client.pull_prompt("langchain-ai/chat-langchain-rephrase", include_model=True)
    #print(prompt)
    stuff_doc_chain  = create_stuff_documents_chain(llm,prompt)

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever=vector_store.as_retriever(), prompt=prompt
    )

    qa =  create_retrieval_chain(
        retriever=history_aware_retriever,combine_docs_chain=stuff_doc_chain
    )
    result = qa.invoke({
        "input": query,
        "chat_history": chat_history,
        "context": stuff_doc_chain  # or retriever.get_relevant_documents(query)
    })
    return result

if __name__ == "__main__":
    res = run_llm(query="why are you looking for a change")
    print(res["answer"])
    

