"""Quick check: is the patents page in the DB and does it contain 'duplexer'?"""
import os
from dotenv import load_dotenv
load_dotenv(override=True)

import psycopg2
from pgvector.psycopg2 import register_vector

dsn = os.getenv("PG_DSN", "postgresql://postgres:postgres@localhost:5433/ecen")
print(f"Connecting to: {dsn}")

try:
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()
    with conn.cursor() as c:
        c.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    register_vector(conn)

    # Check if patents page is indexed
    cur.execute("SELECT chunk_id, LEFT(text, 500) FROM ecen_docs WHERE url LIKE '%patents%'")
    rows = cur.fetchall()
    print(f"\nPatents page chunks in DB: {len(rows)}")
    for r in rows:
        print(f"chunk_id={r[0]}\n{r[1]}\n---")

    # Check if 'duplexer' appears anywhere
    cur.execute("SELECT chunk_id, url, LEFT(text, 300) FROM ecen_docs WHERE text ILIKE '%duplexer%'")
    rows2 = cur.fetchall()
    print(f"\nChunks containing 'duplexer': {len(rows2)}")
    for r in rows2:
        print(f"  url={r[1]}\n  {r[2]}\n")

    conn.close()
except Exception as e:
    print(f"Error: {e}")
