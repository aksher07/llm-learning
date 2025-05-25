import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={"prompt": "Write a poem about open source ML"},
    stream=True
)

for chunk in response.iter_content(chunk_size=None):
    print(chunk.decode(), end="")