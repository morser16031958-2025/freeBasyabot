import asyncio
import sys
sys.path.insert(0, '.')
import webapp
from fastapi.testclient import TestClient

client = TestClient(webapp.app)

print("=== Test SSE endpoint ===")
resp = client.post("/api/chat/stream", json={
    "messages": [{"role": "user", "content": "привет, ответь одним словом"}],
    "model": "nemotron-3-ultra",
    "user_id": 12345,
})
print(f"Status: {resp.status_code}")
print(f"Content-Type: {resp.headers.get('content-type')}")

# Parse SSE events
tokens = []
for line in resp.text.split('\n'):
    if line.startswith('data: '):
        import json
        data = json.loads(line[6:])
        if 'token' in data:
            tokens.append(data['token'])
        if 'error' in data:
            print(f"Error: {data['error']}")
        if data.get('done'):
            print("Done received")

full = ''.join(tokens)
print(f"Tokens: {len(tokens)}, Total: {len(full)} chars")
print(f"Response: {full[:200]}")
