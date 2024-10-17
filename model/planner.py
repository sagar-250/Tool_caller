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
load_dotenv()

os.getenv("GROQ_API_KEY")

class CustomMultiQuery():
    def __init__(self):
        self.llm=None
    
    def model_post_init(self) -> None:
        """
        This method is automatically called after the model is initialized.
        It sets up the custom prompt, output parser, LLM chain, and retriever.
        """
        # Define the expected response schema
        response_schemas = [
            ResponseSchema(
                name="steps",
                description="A list of steps breaking down the original question."
            )
        ]

        # Create the output parser
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

        # Get format instructions
        format_instructions = output_parser.get_format_instructions()

        # Customized prompt to consider argument descriptions and generate accurate tool-based steps
        prompt_template = """Your task is to break down the question into a list of different steps.
The steps should be as specific as possible, using the following tool descriptions and argument details.
Your goal is to have the task broken into the minimum number of simpler atomic steps.

Tool Descriptions:
{tool_descriptions}
Use `search_object_by_name` only when the question mentions searching for any object; else use `who_am_i`.
Example question: Prioritize my P0 issues and add them to the current sprint
Provide the answer in the following format:
{format_instructions}

if not solvable return EMPTY  LIST
Now solve the following question
Original question: {question}"""

        prompt_obj = PromptTemplate(
            input_variables=["question", "tool_descriptions"],
            partial_variables={"format_instructions": format_instructions},
            template=prompt_template,
        )
        
        llm = ChatGroq(
        model="llama-3.2-90b-vision-preview",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        # LLM Chain combining prompt and output parsing
        self.llm_chain = LLMChain(
            llm=llm,
            prompt=prompt_obj,
            output_parser=output_parser
        )

    def planner(self, query: str,description:list) -> List[str]:
        """
        Breaks down the user's query into step-by-step actions based on the tool descriptions.
        """
        # Use the LLM chain to process the query with tool descriptions
        
        try:
            parsed_output = self.llm_chain.predict(
                question=query,
                tool_descriptions=description
            )
            # Extract the steps from the parsed output
            steps = parsed_output['steps']
            # print(steps)
            return steps
        except:
            return "Not-Possible"

# mock_interiit_tools = [
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
#       "tool_name": "works_list",
#       "tool_description": "Returns a list of work items matching the request",
#       "args": [
#           {
#               "arg_name": "applies_to_part",
#               "arg_type": "str",
#               "is_array": True,
#               "is_required": False
#           },
#           {
#               "arg_name": "created_by",
#               "arg_type": "str",
#               "is_array": True,
#               "is_required": False
#           },
#           {
#               "arg_name": "issue_priority",
#               "arg_type": "str",
#               "is_array": True,
#               "is_required": False
#           },
#           {
#               "arg_name": "issue.rev_orgs",
#               "arg_type": "str",
#               "is_array": True,
#               "is_required": False
#           },
#           {
#               "arg_name": "limit",
#               "arg_type": "int",
#               "is_array": False,
#               "is_required": False
#           },
#           {
#               "arg_name": "owned_by",
#               "arg_type": "str",
#               "is_array": True,
#               "is_required": False
#           },
#           {
#               "arg_name": "stage_name",
#               "arg_type": "str",
#               "is_array": True,
#               "is_required": False
#           },
#           {
#               "arg_name": "ticket_need_response",
#               "arg_type": "boolean",
#               "is_array": False,
#               "is_required": False
#           },
#           {
#               "arg_name": "ticket_rev_org",
#               "arg_type": "str",
#               "is_array": True,
#               "is_required": False
#           },
#           {
#               "arg_name": "ticket_severity",
#               "arg_type": "str",
#               "is_array": True,
#               "is_required": False
#           },
#           {
#               "arg_name": "ticket_source_channel",
#               "arg_type": "str",
#               "is_array": True,
#               "is_required": False
#           },
#           {
#               "arg_name": "type",
#               "arg_type": "str",
#               "is_array": True,
#               "is_required": False,
#               "arg_description":"Filters for work ofthe provided types. Allowed values: issue, ticket, task "}
#       ],
#       "output": {
#           "arg_type": "any",
#           "is_array": False,
#           "is_required": True
#       }
#   },
#   {
#       "tool_name": "summarize_objects",
#       "tool_description": "Summarizes a list of objects. The logic of how to summarize a particular object type is an internal implementation detail.",
#       "args": [
#           {
#               "arg_name": "objects",
#               "arg_type": "any",
#               "is_array": True,
#               "is_required": True
#           }
#       ],
#       "output": {
#           "arg_type": "any",
#           "is_array": False,
#           "is_required": True
#       }
#   },
#   {
#       "tool_name": "prioritize_objects",
#       "tool_description": "Returns a list of objects sorted by priority. The logic of what constitutes priority for a given object is an internal implementation detail",
#       "args": [
#           {
#               "arg_name": "objects",
#               "arg_type": "any",
#               "is_array": False,
#               "is_required": False
#           }
#       ],
#       "output": {
#           "arg_type": "any",
#           "is_array": True,
#           "is_required": True
#       }
#   },
#   {
#       "tool_name": "add_work_items_to_sprint",
#       "tool_description": "Adds the given work items to the sprint",
#       "args": [
#           {
#               "arg_name": "work_ids",
#               "arg_type": "str",
#               "is_array": True,
#               "is_required": True
#           },
#           {
#               "arg_name": "sprint_id",
#               "arg_type": "str",
#               "is_array": False,
#               "is_required": True
#           }
#       ],
#       "output": {
#           "arg_type": "boolean",
#           "is_array": False,
#           "is_required": True
#       }
#   },
#   {
#       "tool_name": "get_similar_work_items",
#       "tool_description": "Returns a list of work items that are similar to the given work item",
#       "args": [
#           {
#               "arg_name": "work_id",
#               "arg_type": "str",
#               "is_array": False,
#               "is_required": True
#           }
#       ],
#       "output": {
#           "arg_type": "str",
#           "is_array": True,
#           "is_required": True
#       }
#   },
#   {
#       "tool_name": "search_object_by_name",
#       "tool_description": "Given a search string, returns the id of a matching object in the system of record. If multiple matches are found, it returns the one where the confidence is highest",
#       "args": [
#           {
#               "arg_name": "query",
#               "arg_type": "str",
#               "is_array": False,
#               "is_required": True
#           }
#       ],
#       "output": {
#           "arg_type": "any",
#           "is_array": False,
#           "is_required": True
#       }
#   },
#   {
#       "tool_name": "create_actionable_tasks_from_text",
#       "tool_description": "Given a text, extracts actionable insights, and creates tasks for them, which are kind of a work item.",
#       "args": [
#           {
#               "arg_name": "text",
#               "arg_type": "str",
#               "is_array": False,
#               "is_required": True
#           }
#       ],
#       "output": {
#           "arg_type": "str",
#           "is_array": True,
#           "is_required": True
#       }
#   }
# ]

query="What are my all issues in the triage stage under part FEAT-123? Summarize them"
def steps(query,tools):
    # query="What are my all issues in the triage stage under part FEAT-123? Summarize them."
    l=CustomMultiQuery()
    l.model_post_init()
    steps=l.planner(query,tools)
    print(steps)
    return steps

# print(steps(query,mock_interiit_tools))