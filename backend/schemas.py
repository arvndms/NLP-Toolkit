from pydantic import BaseModel

class TextRequest(BaseModel):
    text:str

class QARequest(BaseModel):
    question:str
    context:str