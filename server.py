from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()


app = FastAPI()

llm = ChatOpenAI(
    openai_api_base= os.getenv("API_BASE"),
    openai_api_key= os.getenv("API_KEY"),  # 🔒 Replace with secure handling
    model="llama_3_3_70b_instruct_fp8",
    streaming=True,
    temperature=0,
)

@app.get("/generate")
# this is the path

async def generate(prompt: str):
    # here prompt is a query parameter
    messages = [HumanMessage(content=prompt)]

    def generate_stream():
        for chunk in llm.stream(messages):
            yield chunk.content

    return StreamingResponse(generate_stream(), media_type="text/plain")
