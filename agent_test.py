from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langchainhub import Client 
from langgraph.prebuilt import create_react_agent
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from promt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

load_dotenv()
import os

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4.1-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7)
#client = Client()
#react_prompt = client.pull("hwchase17/react")
output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
react_prompt_with_format_inst = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["tools", "input", "agent_scratchpad"],
    partial_variables={
        "format_instructions": output_parser.get_format_instructions(),
        "input": "search for 3 jobs related to AI engineer in HYderabad and summarize them for me",
        "tools": "functions.tavily_search",
        "agent_scratchpad": "{agent_scratchpad}"
        
    }
).partial(format_instruction=output_parser.get_format_instructions())
#print(react_prompt)
agent = create_react_agent(model=llm, tools=tools, prompt=react_prompt_with_format_inst,debug=True)
#agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
#chain = agent_executor

def main():
    result = agent.invoke(
        {"input": "search for 3 jobs related to AI engineer in HYderabad and summarize them for me"}
    )
    print(result)

if __name__ == "__main__":
    main()  