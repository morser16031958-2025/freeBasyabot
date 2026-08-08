import asyncio
import sys
sys.path.insert(0, '.')
import provider

async def test():
    # Test 1: simple question
    print("=== Test 1: simple question ===")
    history = [{'role': 'user', 'content': 'привет, ответь одним словом'}]
    chunks = []
    try:
        async for chunk in provider.ask_stream(history, model='nemotron-3-ultra'):
            chunks.append(chunk)
        full = ''.join(chunks)
        print(f"OK: {len(chunks)} chunks, {len(full)} chars")
        print(f"Response: {full[:200]}")
    except Exception as e:
        print(f"FAIL: {e}")

    # Test 2: question that needs web search
    print("\n=== Test 2: web search question ===")
    history = [{'role': 'user', 'content': 'какой сегодня курс доллара в рублях?'}]
    chunks = []
    try:
        async for chunk in provider.ask_stream(history, model='nemotron-3-ultra'):
            chunks.append(chunk)
        full = ''.join(chunks)
        print(f"OK: {len(chunks)} chunks, {len(full)} chars")
        print(f"Response: {full[:300]}")
    except Exception as e:
        print(f"FAIL: {e}")

asyncio.run(test())
