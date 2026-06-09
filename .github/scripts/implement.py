#!/usr/bin/env python3
"""
implement.py — Called by codex.yml to generate and apply a fix.
Reads the issue, calls the OpenAI API, and writes changes directly
to the runner filesystem so git can capture them.
"""

import json
import os
import sys
from openai import OpenAI

BACKEND_FILES = [
    "backend/generator.py",
    "backend/retriever.py",
    "backend/main.py",
    "backend/graph_retriever.py",
]


def read_files() -> str:
    parts = []
    for path in BACKEND_FILES:
        try:
            with open(path) as f:
                parts.append(f"=== {path} ===\n{f.read()}")
        except FileNotFoundError:
            pass
    return "\n\n".join(parts)


def main() -> None:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    issue_number = os.environ.get("ISSUE_NUMBER", "?")
    issue_title  = os.environ.get("ISSUE_TITLE", "")
    issue_body   = os.environ.get("ISSUE_BODY", "")
    comment_body = os.environ.get("COMMENT_BODY", "")

    files_content = read_files()

    prompt = f"""You are fixing a bug in the TAMU ECE RAG chatbot (FastAPI/Python backend + Next.js frontend).

Issue #{issue_number}: {issue_title}

Issue body:
{issue_body}

Approval comment:
{comment_body}

Current backend files:
{files_content}

Make the SMALLEST correct fix. Rules:
- Only change what is needed to fix the reported issue.
- Do not break imports, startup, the retrieval pipeline, or answer completeness.
- Keep every Python file py_compile-clean.
- Do not change unrelated behaviour.

Output a JSON object with EXACTLY this structure (no markdown, no code fences):
{{
  "summary": "One sentence describing root cause and fix",
  "files": [
    {{
      "path": "backend/generator.py",
      "content": "...complete new file content..."
    }}
  ]
}}

Only include files that actually need to change."""

    print(f"Calling OpenAI API for issue #{issue_number}…")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=8000,
        temperature=0.1,
    )

    raw = response.choices[0].message.content
    print(f"Response received ({len(raw)} chars)")

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}\n{raw[:500]}")
        sys.exit(1)

    print(f"Summary: {result.get('summary', '')}")

    files = result.get("files", [])
    print(f"Files to change: {[f['path'] for f in files]}")

    for fc in files:
        path    = fc.get("path", "").strip()
        content = fc.get("content", "")
        if not path or not content:
            continue
        # Safety: only allow source directories
        if not any(path.startswith(d) for d in ("backend/", "crawler/", "frontend/")):
            print(f"Skipping disallowed path: {path}")
            continue
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        print(f"Written: {path}")

    if not files:
        print("No files changed.")


if __name__ == "__main__":
    main()
