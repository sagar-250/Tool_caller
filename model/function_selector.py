import json
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import ast


class EnhancedVectorStoreRetriever:
    
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
            print(doc)
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
    def get_docs(self):
        return self.documents
    def get_retriever(self):
        return self.vector_store




    
def BM25_search(query, tokenized_corpus):

    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = word_tokenize(query.lower())


    doc_scores = bm25.get_scores(tokenized_query)

 
    top_10_i = sorted(
        range(len(doc_scores)), 
        key=lambda i: doc_scores[i], 
        reverse=True
    )[:10]
    top_10_docs = [tokenized_corpus[i] for i in top_10_i]

    return top_10_docs


    

def union_by_toolname(list1, list2):

    merged_dict = {item['tool_name']: item for item in list1}
    merged_dict.update({item['tool_name']: item for item in list2})
    merged_list = list(merged_dict.values())
    return merged_list    


def tool_retriever(query,tools):
    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs_retriever = EnhancedVectorStoreRetriever(hf_embeddings, "my_vector_store", tools)
    tokenized_corpus = [str(chunk) for chunk in tools]
    retrieved_docs_cos = vs_retriever.get_retriever().similarity_search(query,k=10)
    retrieved_tools_cos=[]
    for i in retrieved_docs_cos:
        retrieved_tools_cos.append(json.loads(i.page_content))

    retrieved_docs_bm=BM25_search(query,tokenized_corpus )
    retrieved_tools_bm=[]
    for i in retrieved_docs_bm:
        retrieved_tools_bm.append(ast.literal_eval(retrieved_docs_bm[0]))
    
    merged_retrived_tool=union_by_toolname(retrieved_tools_cos,retrieved_tools_bm)
    print(len(merged_retrived_tool))
    print(type(merged_retrived_tool))
    return merged_retrived_tool
    


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