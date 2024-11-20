import requests

url = "http://localhost:8100/predict"
payload = {"url": 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQTCB6VnFX5hdjp0VFL3ZcWfUuRspElY1nUXQ&s'}
response = requests.post(url, json=payload)
print(response.json())
