import pandas as pd
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
# from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.tools import tool
# from langchain_core.messages import HumanMessage, ToolMessage
# from langchain.agents import initialize_agent,AgentType
import yaml
import pathlib

load_dotenv()

os.getenv("GROQ_API_KEY")

root_dir = pathlib.Path(__file__).parent
print(root_dir)
prompt_file_path=str(root_dir /'prompt'/'prompt.yaml' )

with open(prompt_file_path, 'r') as file:
    prompts = yaml.safe_load(file)['Prompts']



print(prompts.keys())


class SceneGenerator:
  
  def __init__(
            self, 
            model_id: str = "llama-3.1-8b-instant",
            max_generation_tokens: int = None,
            max_retries: int = 5,
            temperature:float=0.3):
    
        self._model_id = model_id
        self._max_generation_tokens = max_generation_tokens
        self._max_retries = max_retries
        self._temperature = temperature
        self._llm = None
    
  def intialize_llm(self):
    
    llm = ChatGroq(
    model=self._model_id,
    temperature=self._temperature,
    max_tokens=self._max_generation_tokens,
    timeout=None,
    max_retries=self._max_retries,
    )
    self._llm=llm
    return self._llm
  
  def prompt_tool_list(self):
    
    @tool
    def laser_pecker_engraving(query:str):  
      """tis function genrates desirable scene tree for laser engraving, takes only one arg of user query"""
      
      system_prompt=prompts["LaserEngraving"]
      system_message_template = SystemMessagePromptTemplate.from_template(system_prompt)
      user_message_template = HumanMessagePromptTemplate.from_template("{user_query}")

      chat_prompt = ChatPromptTemplate.from_messages([system_message_template, user_message_template])

      llm_chain = LLMChain(llm=self._llm , prompt=chat_prompt)

      response = llm_chain.run(user_query=query)
      
      self._response=response
      
      with open("response.txt", "w") as file:
              file.write(response)
      print(response)
      return """ Scene tree sucessfully generated you can teel the task is complete """ 
    
    @tool
    def autoscrew(query:str):  
      """tis function genrates desirable scene tree for auto screwing, takes only one arg of user query"""
  
      system_prompt=prompts["Autoscrewing"]
    
      system_message_template = SystemMessagePromptTemplate.from_template(system_prompt)
      user_message_template = HumanMessagePromptTemplate.from_template("{user_query}")
      chat_prompt = ChatPromptTemplate.from_messages([system_message_template, user_message_template])
      llm_chain = LLMChain(llm=self._llm , prompt=chat_prompt)
      response = llm_chain.run(user_query=query)      
      self._response=response 
      with open("response.txt", "w") as file:
              file.write(response)
      print(response)
      return """ Scene tree sucessfully generated """
    
    @tool
    def paletizing(query:str):  
      """tis function genrates desirable scene tree for paletizing, takes only one arg of user query"""
      
      system_prompt=prompts["Palletizing"]
      
      system_message_template = SystemMessagePromptTemplate.from_template(system_prompt)
      user_message_template = HumanMessagePromptTemplate.from_template("{user_query}")
      chat_prompt = ChatPromptTemplate.from_messages([system_message_template, user_message_template])
      llm_chain = LLMChain(llm=self._llm , prompt=chat_prompt)
      response = llm_chain.run(user_query=query)      
      self._response=response      
      with open("response.txt", "w") as file:
              file.write(response)
      print(response)
      return """ Scene tree sucessfully generated """
    
    @tool
    def CNC_machine_tending(query:str):  
      """this function genrates desirable scene tree for CNC_machine_tending, takes only one arg of user query"""
      
      system_prompt=prompts["CNC"]

      system_message_template = SystemMessagePromptTemplate.from_template(system_prompt)
      user_message_template = HumanMessagePromptTemplate.from_template("{user_query}")

      chat_prompt = ChatPromptTemplate.from_messages([system_message_template, user_message_template])

      llm_chain = LLMChain(llm=self._llm , prompt=chat_prompt)

      response = llm_chain.run(user_query=query)
      
      self._response=response
      
      with open("response.txt", "w") as file:
              file.write(response)
      print(response)
      return """ Scene tree sucessfully generated """
    
    self._tool_list=[paletizing,autoscrew,CNC_machine_tending,laser_pecker_engraving]
    self._tool_dict={"paletizing":paletizing,"autoscrew":autoscrew,"CNC_machine_tending":CNC_machine_tending,"laser_pecker_engraving":laser_pecker_engraving}
    
    return self._tool_list,self._tool_dict
  
  def tool_calling(self, query):
    llm_tools = self._llm.bind_tools(self._tool_list)
    ai_msg = llm_tools.invoke(query)
    print(ai_msg)
    for tool_call in ai_msg.tool_calls:
        selected_tool = self._tool_dict[tool_call["name"].lower()]
        tool_output = selected_tool.invoke(query)
    
    print(tool_output)    
        
        
    

if __name__ == "__main__":
    llm_planner = SceneGenerator()
    llm_planner.intialize_llm()
    tool_list=llm_planner.prompt_tool_list()
    query="make a scene for auto screwing having two cardboard a robot mounter with gripper and collection of screws"
    llm_planner.tool_calling(query)
    
 