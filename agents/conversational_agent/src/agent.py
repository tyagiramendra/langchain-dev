import os
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, ChatMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from pydantic import BaseModel, Field
from langchain.memory import ConversationBufferMemory

from langgraph.checkpoint.memory import MemorySaver  
from langgraph.prebuilt import create_react_agent

from src.tools import search_wikipedia,search_google
memory = ConversationBufferMemory(return_messages=True,memory_key="chat_history")

class ConversationAgent():
    def __init__(self) -> None:
        self.tools  = [search_google, search_wikipedia]
        self.functions = [format_tool_to_openai_function(f) for f in self.tools]
        self.model = ChatOpenAI(temperature=0).bind(functions=self.functions)

    def execute(self,query:str):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are helpful but sassy assistant"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        agent_chain = RunnablePassthrough.assign(
            agent_scratchpad= lambda x: format_to_openai_functions(x["intermediate_steps"])
        ) | prompt | self.model | OpenAIFunctionsAgentOutputParser()

        agent_executor = AgentExecutor(agent=agent_chain, tools=self.tools,  verbose=True, memory=memory)
        respose =agent_executor.invoke({"input":query})
        return respose["output"]