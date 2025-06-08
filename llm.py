import asyncio
import httpx
import time
import sys
import json


async def oai_compat(url, fname):
    with open(fname) as f:
        messages = json.load(f)
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", url, json=messages, timeout=None) as response:
            if response.status_code == 200:
                start_time = time.time()
                count = 0
                async for chunk in response.aiter_bytes():
                    try:
                        temp = json.loads(chunk.decode().replace('\n', '', 1)[6:])
                        token = temp["choices"][0]["delta"]["content"]
                        if token is not None:
                            print(token, end="", flush=True)
                        if "usage" in temp:
                            print("usage: ", temp["usage"], file=sys.stderr, flush=True)
                    except Exception:
                        print(temp, file=sys.stderr, flush=True)
                    count += 1
                    end_time = time.time()
                duration = end_time - start_time
                print(f"Received {count} chunks in {duration:.2f} seconds", file=sys.stderr)
            else:
                print(f"Error: {response.status_code} - {(await response.aread()).decode()}",
                      file=sys.stderr)


if __name__ == '__main__':
    url = sys.argv[1]
    messages = sys.argv[2]
    asyncio.run(oai_compat(url, messages))
