import json
import sys
sys.path.insert(0, '.')
import webapp
from fastapi.testclient import TestClient

client = TestClient(webapp.app)

print("=== Test SSE with web search ===")
resp = client.post("/api/chat/stream", json={
    "messages": [{"role": "user", "content": "какой сегодня курс доллара?"}],
    "model": "nemotron-3-ultra",
    "user_id": 12345,
})
print(f"Status: {resp.status_code}")

tokens = []
errors = []
for line in resp.text.split('\n'):
    if line.startswith('data: '):
        data = json.loads(line[6:])
        if 'token' in data:
            tokens.append(data['token'])
        if 'error' in data:
            errors.append(data['error'])
        if data.get('done'):
            print("Done received")

if errors:
    print(f"Errors: {errors}")
else:
    full = ''.join(tokens)
    print(f"Tokens: {len(tokens)}, Total: {len(full)} chars")
    print(f"Response preview: {full[:300]}")
