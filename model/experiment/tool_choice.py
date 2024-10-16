import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import Tool
from langchain.agents import initialize_agent,AgentType
from langchain_core.tools import tool

# from langchain.chains import SequentialChain


# Load environment variables from the .env file
load_dotenv()

os.getenv("GROQ_API_KEY")




tools=[
  {
      "tool_name": "who_am_i",
      "tool_description": "Returns the id of the current user",
      "args": [],
      "output": {
          "arg_type": "str",
          "is_array": False,
          "is_required": True
      }
  },
  {
      "tool_name": "get_sprint_id",
      "tool_description": "Returns the ID\nof the current\nsprint",
      "args": [],
      "output": {
          "arg_type": "str",
          "is_array": False,
          "is_required": True
      }
  }]

llm = ChatGroq(
 model="llama-3.1-8b-instant",
temperature=0.3,
max_tokens=None,
timeout=None,
max_retries=2,
tools=tools,
tool_choice="auto",
)

# llm.run("hi")

@tool
def add(num1:int,num2:int):
    """this function returns the addition of given two numbers"""
    return(num1+num2)
@tool
def subtraction(num1:int,num2:int):
    """this function returns the subtraction of given two numbers"""
    return(num1-num2)



tools=[add,subtraction]
llm_tools = llm.bind_tools(tools)


# output=llm_tools.invoke(prompt)

# print("\n---Tool Calls---")
# print(output.tool_calls)

query = "addition 1 and 2 "

messages = [HumanMessage(query)]
ai_msg = llm_tools.invoke(messages)
print()
messages.append(ai_msg)

# Only using web search and exchange rate, and the Pydantic schema is not a full function, just a container for arguments
for tool_call in ai_msg.tool_calls:
    selected_tool = {"add": add, "subtraction": subtraction}[tool_call["name"].lower()]
    tool_output = selected_tool.invoke(tool_call["args"])
    print(tool_output)