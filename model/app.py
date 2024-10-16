from fastapi import FastAPI,File, UploadFile
from fastapi.responses import JSONResponse
from typing import Optional
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pathlib
import os
import uvicorn 
import json
import PyPDF2
from tool_caller import output

root_dir = pathlib.Path(__file__).parent

pdf_path=str(root_dir /'data'/'en'/'extra.pdf' )
txt_path=str(root_dir /'data'/'en'/'extra.txt' )


app= FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query1(BaseModel):
    query:str
    tools:str

 
@app.post('/toolcalling')
def get_response(req:Query1):
    user_prompt=req.query
    tools=json.loads(req.tools)
    tools.append({
    "tool_name": "NOT_POSSIBLE",
    "tool_description": "USED WHEN THE OTHER TOOLS ARE NOT SUFFICIENT OR CANT DO THE TASK",
    "args": [],
    "output": {
    "arg_type": "any",
    "is_array": True,
    "is_required": True
    }
    })
    tool_calls=output(user_prompt,tools)
    for tool in tool_calls:
        tool=json.loads(tool)
        if tool["tool_name"] == "NOT_POSSIBLE":
            return []
    return tool_calls



uvicorn.run(app,host="127.0.0.1",port=8000)        