import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS



load_dotenv()

if __name__ == "__main__":
    google_api_key = os.getenv("GOOGLE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    index_name = os.getenv("INDEX_NAME")

    print("Google API Key:", google_api_key)
    print("OpenAI API Key:", openai_api_key)
    print("Index Name:", index_name)        

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # vector_store = PineconeVectorStore.from_existing_index(
    #     index_name=index_name,
    #     embedding=embedding
    # )

    llm = ChatOpenAI(model="gpt-4.1-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7)
    #retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template("""
        Answer the question based only on the following context:

        Context: {context}

        Question: {question}

        Provide a detailed answer based on the context above. If the answer cannot be found in the context, 
        say "I don't have enough information to answer this question."
        """)
    

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Build the RAG chain manually (langchain 1.0+ style)
    # rag_chain = (
    #     {
    #         "context": retriever | format_docs,
    #         "question": RunnablePassthrough()
    #     }
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )

   # Run the query
    # question = "What is SUman total experience?"
    # print(f"\nQuestion: {question}")

    # result = rag_chain.invoke(question)
    # print("\nResult:", result)

    vector_store_faiss = FAISS.load_local(
        "faiss_index_rag",embedding,allow_dangerous_deserialization=True
    )
    retriever_faiss = vector_store_faiss.as_retriever(search_kwargs={"k": 3})
    rag_chain_faiss = (
        {
            "context": retriever_faiss | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

   # Run the query
    question = "What is Suman total experience?"
    print(f"\nQuestion: {question}")

    result = rag_chain_faiss.invoke(question)
    print("\nResult:", result)