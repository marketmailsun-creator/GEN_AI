from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from langchain_community.cache import SQLiteCache as sql
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableMap
import langchain
from dotenv import load_dotenv
import os
#Load environment variables
load_dotenv()

#lCreate an LLM instance
langchain.debug = True  #Enable debug mode to see detailed logs
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7
)


#Create messages OBJECT 
messages = [
    SystemMessage(content="You are a helpful assistant that translates English to French."),
    HumanMessage(content="Translate the following English text to French: 'Hello, how are you'")
]

#INVOKE THE LLM WITH THE MESSAGES OR A STRING PROMPT
'''response = llm.invoke("Say hello from Hyderabad!")
response = llm.invoke(messages)
print(response.text)'''

#Set up caching (optional)
#set_llm_cache(InMemoryCache())  # or SQLiteCache(database_path="llm_cache.db")
'''set_llm_cache(sql(database_path=".langchain.db"))
response = llm.invoke("Tell me a joke ") #This call will be cached
print(response.text)
response1=llm.invoke("Tell me a joke ") # This call will be faster due to caching
print(response1.text)'''

###
#Streaming response : It shows token by token generation
###
'''for chunk in llm.stream("Explain the theory of relativity in brief."):
    print(chunk.text, end='', flush=True)'''

#Using PromptTemplate, Used for simple prompts with variable substitutions
'''template = "You are a virologist, explain the {topic} in {language} language."
prompt_template = PromptTemplate.from_template(template=template)
prompt = prompt_template.format(topic="COVID-19", language="English")
print(llm.invoke(prompt).text)'''


#Using ChatPromptTemplate
# used for more complex prompts with multiple messages
'''chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="You provide output only in json format."),
    HumanMessagePromptTemplate.from_template("Provide top {n} contries of {area} by hapiness index.")

])
messages = chat_template.format_messages(n=5, area="Asia")
response = llm.invoke(messages)
print(response.text)'''

#Using LLMChain simple chain with prompt template and LLM
'''template = 'you are a travel guide. Suggest a travel destination in {season} for a {type_of_traveler}.'
prompt = PromptTemplate.from_template(template)
chain = prompt | llm | StrOutputParser()  #LLMChain(llm=llm, prompt=prompt, verbose=True)
season = input("Enter the season (e.g., summer, winter): ")
type_of_traveler = input("Enter the type of traveler (e.g., solo, family, adventure seeker): ")
response = chain.invoke(
    {
     "season": season, 
     "type_of_traveler": type_of_traveler
    }
)
print(response)'''

#Using sequential chains it can use mulipple llms and prompt templates in a sequence

prompt_template1 = PromptTemplate.from_template("you are a experienced scientist and python developer. Write a function that implements about {concept}.")
prompt_template2 = PromptTemplate.from_template("Given the following python function {function}, explain it as detailed as possible for a beginner in properly formatted markdown.")

function_chain = prompt_template1 | llm | StrOutputParser() #|prompt_template2 | llm | StrOutputParser()
details_chain = prompt_template2 | llm | StrOutputParser()

topic = input("Enter a programming topic: ")
sequential_chain = RunnableSequence(
        {
            "function": function_chain.bind(),
        }, 
        {
            "details": details_chain.bind(),
        },
)

def pipeline(input_topic):
    function = function_chain.invoke({"concept": input_topic})
    details = details_chain.invoke({"function": function})
    return {"function": function, "details": details}

result = pipeline(topic)
print("Function:", result["function"])
print("Details of fucntion:", result["details"])



