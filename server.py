from fastapi import FastAPI, Header, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for production, set your frontend origin explicitly
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


API_KEY = "supersecretapikey" 

llm = ChatOpenAI(
    openai_api_base="https://infer.e2enetworks.net/project/p-5729/genai/llama_3_3_70b_instruct_fp8/v1",
    openai_api_key="eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJGSjg2R2NGM2pUYk5MT2NvNE52WmtVQ0lVbWZZQ3FvcXRPUWVNZmJoTmxFIn0.eyJleHAiOjE3NzkyNzQ0MzQsImlhdCI6MTc0NzczODQzNCwianRpIjoiY2QzZTk2NDgtZDk5Ny00MmNhLTg2YmMtYjRiMDQ1MzdlMWFkIiwiaXNzIjoiaHR0cDovL2dhdGV3YXkuZTJlbmV0d29ya3MuY29tL2F1dGgvcmVhbG1zL2FwaW1hbiIsImF1ZCI6ImFjY291bnQiLCJzdWIiOiIwZGMyZGE0ZC1iYzk4LTQ3ZmEtOWY4My01NjcyZmMyMWVmZWIiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJhcGltYW51aSIsInNlc3Npb25fc3RhdGUiOiIxY2ZmODJiYS03ZGI5LTQyZjktODFjNy02Yzg4ODA3ZGZhNTQiLCJhY3IiOiIxIiwiYWxsb3dlZC1vcmlnaW5zIjpbIiJdLCJyZWFsbV9hY2Nlc3MiOnsicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiIsImFwaXVzZXIiLCJkZWZhdWx0LXJvbGVzLWFwaW1hbiJdfSwicmVzb3VyY2VfYWNjZXNzIjp7ImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJtYW5hZ2UtYWNjb3VudC1saW5rcyIsInZpZXctcHJvZmlsZSJdfX0sInNjb3BlIjoicHJvZmlsZSBlbWFpbCIsInNpZCI6IjFjZmY4MmJhLTdkYjktNDJmOS04MWM3LTZjODg4MDdkZmE1NCIsImVtYWlsX3ZlcmlmaWVkIjpmYWxzZSwiaXNfcGFydG5lcl9yb2xlIjpmYWxzZSwibmFtZSI6IkFudXNoa2EgUmFoZWphIiwicHJpbWFyeV9lbWFpbCI6ImFudXNoa2EucmFoZWphQGUyZW5ldHdvcmtzLmNvbSIsImlzX3ByaW1hcnlfY29udGFjdCI6dHJ1ZSwicHJlZmVycmVkX3VzZXJuYW1lIjoiYW51c2hrYS5yYWhlamFAZTJlbmV0d29ya3MuY29tIiwiZ2l2ZW5fbmFtZSI6IkFudXNoa2EiLCJmYW1pbHlfbmFtZSI6IlJhaGVqYSIsImVtYWlsIjoiYW51c2hrYS5yYWhlamFAZTJlbmV0d29ya3MuY29tIiwiaXNfaW5kaWFhaV91c2VyIjpmYWxzZX0.LRTXhbKQ_luHvVwWBbBLMmu5MSmlhdy5ERU437XUOV37_z6Ld-1E98qeh6W5HkTrhkiw2oWmy5XPlsak1KCRcLjW38b8QUKIgFPQVh0h8VNVDwDD1ZhsxK7mknZauS9MDY43oowDa6_rjazJtJ8sTSk9fy0sRBljYON_6rO7xkY",
    model="llama_3_3_70b_instruct_fp8",
    streaming=True,
    temperature=0,
)

class PromptRequest(BaseModel):
    prompt: str

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )

@app.post("/generate")
async def generate(data: PromptRequest, x_api_key: str = Header(...)):
    verify_api_key(x_api_key)
    messages = [HumanMessage(content=data.prompt)]

    def generate_stream():
        for chunk in llm.stream(messages):
            yield chunk.content

    return StreamingResponse(generate_stream(), media_type="text/plain")
