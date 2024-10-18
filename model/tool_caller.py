import pandas as pd
import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.tools import Tool
from langchain.agents import initialize_agent,AgentType
from langchain_core.tools import tool
import instructor
from pydantic import BaseModel
from typing import List
# from langchain.chains import SequentialChain


# Load environment variables from the .env file
load_dotenv()

os.getenv("GROQ_API_KEY")

import json
from groq import Groq
import os

# Initialize Groq client
client = Groq()


class Argument(BaseModel):
    argument_name: str
    argument_value: List[str]

class Tool(BaseModel):
    tool_name: str
    arguments: List[Argument]
    
class ResponseModel(BaseModel):
    tool_calls: list[Tool]
    
client = instructor.from_groq(Groq(), mode=instructor.Mode.JSON) 

def run_conversation(user_prompt,tools):
    tools=tools
    # Prepare the messages
    messages = [
        {
            "role": "system",
            "content": f"""
            **Instructions**:
            - Analyze the given steps, which include tool names and arguments.
            - Use these steps to generate the sequence of tool calls, ensuring correct ordering and dependency management.     
            
                   
            You have access to the following tool: {tools}.
            go through each argument and description if  provided properly and give every needed arg
            
            when you need output from of i th previous tool for argument write "$$PREV[i]" in the argument .order the function calling properly for the task.
            
            
            """
        },
        {
            "role": "user",
            "content": user_prompt,
        }
    ]

    
    
    # Make the Groq API call
    response = client.chat.completions.create(
        model="llama-3.2-90b-vision-preview",
        response_model=ResponseModel,
        messages=messages,
        temperature=0.9,
        max_tokens=1000,
    )

    return response.tool_calls

# Example usage
def validator_crrection(tools,llm_output,validation_message):
    validation_message=f"""**Instructions**:
        - edit the llm_output:{llm_output} according to the validation message:{validation_message}
        - Ensure that all required arguments are present and correctly formatted. 
        - Check for any inconsistencies or missing information in the tool calls.
        
        You have access to the following tools: {tools}.
        Go through each argument and description carefully to confirm they are provided correctly and all necessary arguments are included.
        
        ."""
        
    response = client.chat.completions.create(
    model="llama-3.2-90b-vision-preview",
    response_model=ResponseModel,
    messages=validation_message,
    temperature=0.9,
    max_tokens=1000,
    )

    return response.tool_calls



def output(query,tools):
    tool_calls = run_conversation(query,tools)
    # print(tool_calls)
    
    output=[] 
    for i in tool_calls:
        output.append(i.json())
        
    return output  


  


# EXAMPLE
tools1=[
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
  },
  {
    "tool_name": "works_list",
    "tool_description": "Returns a list of work items matching the request",
    "args": [
    {
    "arg_name": "applies_to_part",
    "arg_type": "str",
    "is_array": True,
    "is_required": False
    },
    {
    "arg_name": "created_by",
    "arg_type": "str",
    "is_array": True,
    "is_required": False
    },
    {
    "arg_name": "issue_priority",
    "arg_type": "str",
    "is_array": True,
    "is_required": False
    },
    {
    "arg_name": "issue.rev_orgs",
    "arg_type": "str",
    "is_array": True,
    "is_required": False
    },
    {
    "arg_name": "limit",
    "arg_type": "int",
    "is_array": False,
    "is_required": False
    },
    {
    "arg_name": "owned_by",
    "arg_type": "str",
    "is_array": True,
    "is_required": False
    },
    {
    "arg_name": "stage_name",
    "arg_type": "str",
    "is_array": True,
    "is_required": False
    },
    {
    "arg_name": "ticket_need_response",
    "arg_type": "boolean",
    "is_array": False,
    "is_required": False
    },
    {
    "arg_name": "ticket_rev_org",
    "arg_type": "str",
    "is_array": True,
    "is_required": False
    },
    {
    "arg_name": "ticket_severity",
    "arg_type": "str",
    "is_array": True,
    "is_required": False
    },
    {
    "arg_name": "ticket_source_channel",
    "arg_type": "str",
    "is_array": True,
    "is_required": False
    },
    {
    "arg_name": "type",
    "arg_type": "str",
    "is_array": True,
    "is_required": False,
    "arg_description":"""Filters for work of
        the provided
        types. Allowed
        values: issue,
        ticket, task"""
    }
    ],
    "output": {
    "arg_type": "any",
    "is_array": False,
    "is_required": True
    }
    },
    {
    "tool_name": "summarize_objects",
    "tool_description": "Summarizes a list of objects. The logic of how to summarize a particular object type is an internal implementation detail.",
    "args": [
    {
    "arg_name": "objects",
    "arg_type": "any",
    "is_array": True,
    "is_required": True
    }
    ],
    "output": {
    "arg_type": "any",
    "is_array": False,
    "is_required": True
    }
    },
    {
    "tool_name": "prioritize_objects",
    "tool_description": "Returns a list of objects sorted by priority. The logic of what constitutes priority for a given object is an internal implementation detail",
    "args": [
    {
    "arg_name": "objects",
    "arg_type": "any",
    "is_array": False,
    "is_required": False
    }
    ],
    "output": {
    "arg_type": "any",
    "is_array": True,
    "is_required": True
    }
}]

user_prompt="Summarize tickets from ’support’ channel"   
L=output(user_prompt,tools1)    
print(L)
    
    