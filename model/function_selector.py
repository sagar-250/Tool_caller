import json
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


class EnhancedVectorStoreRetriever:
    '''
    VectorStoreRetriever customized to handle the new JSON format of tool descriptions,
    including detailed argument descriptions from an external dictionary.
    '''
    def __init__(self, embeddings, name: str, tool_list: list[dict], arg_description_dict: dict=None):
        self.embeddings = embeddings
        self.tool_list = tool_list
        # self.arg_description_dict = arg_description_dict
        self.documents = []

        # Convert tool_list to Document objects, including argument descriptions from the external dictionary of args descriptions
        for i, tool in enumerate(tool_list):
            # Create a detailed representation of each tool
            tool_representation = {
                "tool_name": tool["tool_name"],
                "tool_description": tool["tool_description"],
                "args": [
                    {
                        "arg_name": arg["arg_name"],
                        "arg_type": arg["arg_type"],
                        "is_array": arg["is_array"],
                        "is_required": arg["is_required"],
                        # Fetch description from external dictionary if available
                        # "arg_description": self.arg_description_dict.get(arg["arg_name"], "")
                    } for arg in tool.get("args", [])
                ],
                "output": tool["output"]
            }

            doc_content = json.dumps(tool_representation, indent=2)
            doc = Document(page_content=doc_content, metadata={"index": i})
            self.documents.append(doc)

        # Create or load a vector store
        try:
            self.vector_store = FAISS.load_local(name, self.embeddings)
        except:
            # If no local store exists, create a new one
            self.vector_store = FAISS.from_documents(
                self.documents,
                self.embeddings
            )
            self.vector_store.save_local("/content/drive/MyDrive/vector_store")

        self.vs_name = name
        



    def get_retriever(self):
        return self.vector_store




def tool_retriever(query,tools):
    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs_retriever = EnhancedVectorStoreRetriever(hf_embeddings, "my_vector_store", tools)
    retrieved_docs = vs_retriever.get_retriever().similarity_search(query,k=10)
    retrieved_tools=[]
    for i in retrieved_docs:
        retrieved_tools.append(json.loads(i.page_content))

    print(retrieved_tools)
    return retrieved_tools
    


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

tool_retriever(user_prompt,tools1)