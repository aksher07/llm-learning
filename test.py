import requests

headers = {
    "Content-Type": "application/json",
    "x-api-key": "supersecretapikey"
}
input=input()
data = {
    "prompt": input
}

response = requests.post("http://localhost:8000/generate", json=data, headers=headers, stream=True)

for chunk in response.iter_content(chunk_size=None):
    print(chunk.decode(), end="")
