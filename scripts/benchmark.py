# scripts/benchmark.py
import time
from fastapi.testclient import TestClient
from app.main import app

payload = {"query":"astronaut stranded on Mars","k":5,"page":1,"per_page":5}

def main():
    with TestClient(app) as client:
        t0 = time.time()
        r1 = client.post("/search", json=payload).json()
        uncached = (time.time() - t0) * 1000

        t1 = time.time()
        r2 = client.post("/search", json=payload).json()
        cached = (time.time() - t1) * 1000

        print(f"Uncached: {uncached:.1f} ms")
        print(f"Cached:   {cached:.1f} ms")
        print([x["title"] for x in r2["results"]])

if __name__ == "__main__":
    main()
