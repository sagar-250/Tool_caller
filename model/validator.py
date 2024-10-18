from typing import Any, List
from pydantic import BaseModel, Field, ConfigDict
from langchain.chains.llm import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.schema import BaseRetriever
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import json
load_dotenv()

os.getenv("GROQ_API_KEY")

#@title helper tools
def tools(op):
  try:
    dicts = [json.loads(i) for i in op]
  except:
    dicts =op
  tools=[]
  for k in dicts:
    tools.append(k['tool_name'])
  return tools
def args(op):
  dicts=op
  sum=0
  sum+=len(dicts["arguments"])
  return sum
def args_toolset(op):
  dicts=op
  sum=0
  return len(dicts)
#both return [false,<wrong tool name>]
def validate_tool_arguments(model_tool, actual_tool):
    """
    Validates the arguments provided by the model's tool against the actual tool definition in tools.json.

    Parameters:
    model_tool (dict): The tool and arguments suggested by the model.
    actual_tool (dict): The correct tool definition from tools.json.

    Returns:
    bool: True if the tool arguments are valid, False otherwise.
    str: Error message if the tool arguments are invalid.
    """

    # Extract actual arguments from the correct tool definition
    actual_args = {arg['arg_name']: arg for arg in actual_tool['args']}

    # Track required arguments
    required_args = {arg['arg_name'] for arg in actual_tool['args'] if arg['is_required']}

    # Validate the arguments provided by the model
    for model_arg in model_tool['arguments']:
        arg_name = model_arg['argument_name']

        # Check if the argument name is valid
        if arg_name not in actual_args:
            return False, f"Invalid argument name: '{arg_name}' in tool '{model_tool['tool_name']}'."

        # Remove this argument from required list if it's present
        if arg_name in required_args:
            required_args.remove(arg_name)

    # After checking all provided arguments, ensure all required arguments are present
    if required_args:
        missing_args = ", ".join(required_args)
        return False, f"Missing required argument(s): {missing_args} in tool '{model_tool['tool_name']}'."

    # If everything is correct
    return True, f"Tool '{model_tool['tool_name']}' arguments are valid."


def get_tool_by_name(tool_name, tools):
        for tool in tools:
            if tool["tool_name"] == tool_name:
                return tool
        return None


def checkargs(tools_json,outputs):###remove this function
  tools_used=tools(output)
  tool_args=0
  for i in tools2:
    if i["tool_name"] in tools_used:
      for j in outputs:
        j=eval(j)
        if(j["tool_name"]==i["tool_name"]):
          if(args_toolset(i["args"])<args(j)):
            return (False,i["tool_name"])
  return True
#checks if the tools used all exist
def check_tools(tools_json,outputs):
  tools_used=tools(outputs)
  tool_list=tools(tools_json)
  for i in tools_used:
    if i not in tool_list:
      return False,f"{i} is not an available tool"
  return True,None





###       Main Function         ########################

def arg_validator(model_response, tools):

  output = []

  for i, tool_response in enumerate(model_response):
    try:
      tool_response = eval(tool_response)
    except:
      pass
    tool_name = tool_response["tool_name"]
    tool = get_tool_by_name(tool_name, tools)

    if not tool:
      output.append(f"Tool '{tool_name}' not found in tools.json.")
      continue

    b, msg = validate_tool_arguments(tool_response, tool)

    if not b:
      output.append(msg)

  if output == []:
    return [True], None

  else:
    return [False], output



def error_check(tool_json,op):
    tooler,message=check_tools(tool_json,op)
    if(not tooler):
        return message
    args_val,mess=arg_validator(op,tool_json)
    if(not args_val):
        return mess
    return "All correct"




# tool=[
#   {
#       "tool_name": "who_am_i",
#       "tool_description": "Returns the id of the current user",
#       "args": [],
#       "output": {
#           "arg_type": "str",
#           "is_array": False,
#           "is_required": True
#       }
#   },
#   {
#       "tool_name": "get_sprint_id",
#       "tool_description": "Returns the ID\nof the current\nsprint",
#       "args": [],
#       "output": {
#           "arg_type": "str",
#           "is_array": False,
#           "is_required": True
#       }
#   },
#   {
#     "tool_name": "works_list",
#     "tool_description": "Returns a list of work items matching the request",
#     "args": [
#     {
#     "arg_name": "applies_to_part",
#     "arg_type": "str",
#     "is_array": True,
#     "is_required": False
#     },
#     {
#     "arg_name": "created_by",
#     "arg_type": "str",
#     "is_array": True,
#     "is_required": False
#     },
#     {
#     "arg_name": "issue_priority",
#     "arg_type": "str",
#     "is_array": True,
#     "is_required": False
#     },
#     {
#     "arg_name": "issue.rev_orgs",
#     "arg_type": "str",
#     "is_array": True,
#     "is_required": False
#     },
#     {
#     "arg_name": "limit",
#     "arg_type": "int",
#     "is_array": False,
#     "is_required": False
#     },
#     {
#     "arg_name": "owned_by",
#     "arg_type": "str",
#     "is_array": True,
#     "is_required": False
#     },
#     {
#     "arg_name": "stage_name",
#     "arg_type": "str",
#     "is_array": True,
#     "is_required": False
#     },
#     {
#     "arg_name": "ticket_need_response",
#     "arg_type": "boolean",
#     "is_array": False,
#     "is_required": False
#     },
#     {
#     "arg_name": "ticket_rev_org",
#     "arg_type": "str",
#     "is_array": True,
#     "is_required": False
#     },
#     {
#     "arg_name": "ticket_severity",
#     "arg_type": "str",
#     "is_array": True,
#     "is_required": False
#     },
#     {
#     "arg_name": "ticket_source_channel",
#     "arg_type": "str",
#     "is_array": True,
#     "is_required": False
#     },
#     {
#     "arg_name": "type",
#     "arg_type": "str",
#     "is_array": True,
#     "is_required": False,
#     "arg_description":"""Filters for work of
#         the provided
#         types. Allowed
#         values: issue,
#         ticket, task"""
#     }
#     ],
#     "output": {
#     "arg_type": "any",
#     "is_array": False,
#     "is_required": True
#     }
#     },
#     {
#     "tool_name": "summarize_objects",
#     "tool_description": "Summarizes a list of objects. The logic of how to summarize a particular object type is an internal implementation detail.",
#     "args": [
#     {
#     "arg_name": "objects",
#     "arg_type": "any",
#     "is_array": True,
#     "is_required": True
#     }
#     ],
#     "output": {
#     "arg_type": "any",
#     "is_array": False,
#     "is_required": True
#     }
#     },
#     {
#     "tool_name": "prioritize_objects",
#     "tool_description": "Returns a list of objects sorted by priority. The logic of what constitutes priority for a given object is an internal implementation detail",
#     "args": [
#     {
#     "arg_name": "objects",
#     "arg_type": "any",
#     "is_array": False,
#     "is_required": False
#     }
#     ],
#     "output": {
#     "arg_type": "any",
#     "is_array": True,
#     "is_required": True
#     }
# }]
# op=[
#     {
#         "tool_name": "who_am_i",
#         "arguments": []
#     },
#     {
#         "tool_name": "works_list",
#         "arguments": [
#             {
#                 "argument_name": "applies_to_part",
#                 "argument_value": [
#                     "FEAT-123"
#                 ]
#             },
#             {
#                 "argument_name": "stage_name",
#                 "argument_value": [
#                     "triage"
#                 ]
#             },
#             {
#                 "argument_name": "owned_by",
#                 "argument_value": [
#                     "$$PREV[0]"
#                 ]
#             }
#         ]
#     },
#     {
#         "tool_name": "summarize_objects",
#         "arguments": [
#             {
#                 "argument_name": "objects",
#                 "argument_value": [
#                     "$$PREV[1]"
#                 ]
#             }
#         ]
#     }
# ]

# l=error_check(tool,op)
# print(l)
