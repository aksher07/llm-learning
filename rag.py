from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 1. Load text documents
loader = TextLoader("example.txt")
docs = loader.load_and_split()

# 2. Use HuggingFace embeddings locally (no OpenAI embeddings)
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# 3. Create Chroma DB from documents
chroma_db = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory="data",
    collection_name="lc_chroma_demo"
)

# 4. Initialize E2E-hosted LLaMA 3 model
llm = ChatOpenAI(
    openai_api_base="https://infer.e2enetworks.net/project/p-5729/genai/llama_3_3_70b_instruct_fp8/v1",
    openai_api_key="eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJGSjg2R2NGM2pUYk5MT2NvNE52WmtVQ0lVbWZZQ3FvcXRPUWVNZmJoTmxFIn0.eyJleHAiOjE3NzkyNzQ0MzQsImlhdCI6MTc0NzczODQzNCwianRpIjoiY2QzZTk2NDgtZDk5Ny00MmNhLTg2YmMtYjRiMDQ1MzdlMWFkIiwiaXNzIjoiaHR0cDovL2dhdGV3YXkuZTJlbmV0d29ya3MuY29tL2F1dGgvcmVhbG1zL2FwaW1hbiIsImF1ZCI6ImFjY291bnQiLCJzdWIiOiIwZGMyZGE0ZC1iYzk4LTQ3ZmEtOWY4My01NjcyZmMyMWVmZWIiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJhcGltYW51aSIsInNlc3Npb25fc3RhdGUiOiIxY2ZmODJiYS03ZGI5LTQyZjktODFjNy02Yzg4ODA3ZGZhNTQiLCJhY3IiOiIxIiwiYWxsb3dlZC1vcmlnaW5zIjpbIiJdLCJyZWFsbV9hY2Nlc3MiOnsicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiIsImFwaXVzZXIiLCJkZWZhdWx0LXJvbGVzLWFwaW1hbiJdfSwicmVzb3VyY2VfYWNjZXNzIjp7ImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJtYW5hZ2UtYWNjb3VudC1saW5rcyIsInZpZXctcHJvZmlsZSJdfX0sInNjb3BlIjoicHJvZmlsZSBlbWFpbCIsInNpZCI6IjFjZmY4MmJhLTdkYjktNDJmOS04MWM3LTZjODg4MDdkZmE1NCIsImVtYWlsX3ZlcmlmaWVkIjpmYWxzZSwiaXNfcGFydG5lcl9yb2xlIjpmYWxzZSwibmFtZSI6IkFudXNoa2EgUmFoZWphIiwicHJpbWFyeV9lbWFpbCI6ImFudXNoa2EucmFoZWphQGUyZW5ldHdvcmtzLmNvbSIsImlzX3ByaW1hcnlfY29udGFjdCI6dHJ1ZSwicHJlZmVycmVkX3VzZXJuYW1lIjoiYW51c2hrYS5yYWhlamFAZTJlbmV0d29ya3MuY29tIiwiZ2l2ZW5fbmFtZSI6IkFudXNoa2EiLCJmYW1pbHlfbmFtZSI6IlJhaGVqYSIsImVtYWlsIjoiYW51c2hrYS5yYWhlamFAZTJlbmV0d29ya3MuY29tIiwiaXNfaW5kaWFhaV91c2VyIjpmYWxzZX0.LRTXhbKQ_luHvVwWBbBLMmu5MSmlhdy5ERU437XUOV37_z6Ld-1E98qeh6W5HkTrhkiw2oWmy5XPlsak1KCRcLjW38b8QUKIgFPQVh0h8VNVDwDD1ZhsxK7mknZauS9MDY43oowDa6_rjazJtJ8sTSk9fy0sRBljYON_6rO7xkY",
    model="llama_3_3_70b_instruct_fp8",
    streaming=True,
    temperature=0,
)

# 5. Build RAG chain
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=chroma_db.as_retriever()
)

# 6. Run your query
query = "tell about vpn"
response = chain.invoke({"query": query})
print(response)

