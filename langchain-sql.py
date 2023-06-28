import mysql.connector
import openai 
from langchain.llms import OpenAI
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.sql_database import SQLDatabase

from dotenv import load_dotenv
import os
load_dotenv()
openai.api_key = ""

db = SQLDatabase.from_uri('mysql+pymysql://dhanush:password@localhost:3306/sr_engage')

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI()

toolkit = SQLDatabaseToolkit(db=db,llm=llm)
print("toolkit")
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=False,
    handle_parsing_errors=True,

)
print(agent_executor.run("thanks"))

