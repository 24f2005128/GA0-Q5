import json
import os
import re
import sys
import traceback
from io import StringIO
from typing import List

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeRequest(BaseModel):
    code: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int]


def execute_python_code(code: str) -> dict:
    """Run the submitted code and capture stdout or its full traceback."""
    old_stdout = sys.stdout
    stdout = StringIO()
    sys.stdout = stdout

    try:
        exec(code, {"__name__": "__main__"})
        return {"success": True, "output": stdout.getvalue()}
    except BaseException:
        return {"success": False, "output": traceback.format_exc()}
    finally:
        sys.stdout = old_stdout


def fallback_error_lines(traceback_text: str, code: str) -> List[int]:
    """Use the submitted-code frame from the traceback when AI is unavailable."""
    frames = re.findall(r'File "<string>", line (\d+)', traceback_text)
    if not frames:
        # SyntaxError tracebacks include the submitted line number as an attribute,
        # but may not include a normal <string> frame.
        match = re.search(r"line (\d+)", traceback_text)
        if match:
            return [int(match.group(1))]
        return []

    line_number = int(frames[-1])
    code_lines = code.splitlines()
    return [line_number] if 1 <= line_number <= len(code_lines) else []


def analyze_error_with_ai(code: str, traceback_text: str) -> List[int]:
    token = os.getenv("AIPIPE_TOKEN")
    fallback = fallback_error_lines(traceback_text, code)
    if not token:
        return fallback

    prompt = (
        "Find the 1-based line number(s) in CODE that caused the Python error. "
        "Use TRACEBACK as evidence. Return only JSON as "
        '{"error_lines": [3]}. Do not include line numbers outside CODE.\n\n'
        f"CODE:\n{code}\n\nTRACEBACK:\n{traceback_text}"
    )

    try:
        response = httpx.post(
            "https://aipipe.org/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "model": "gpt-4.1-nano",
                "messages": [
                    {"role": "system", "content": "Return only a JSON object with an error_lines integer array."},
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
            },
            timeout=20,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        result = ErrorAnalysis.model_validate(json.loads(content))
        line_count = len(code.splitlines())
        valid_lines = sorted({n for n in result.error_lines if 1 <= n <= line_count})
        return valid_lines or fallback
    except Exception:
        return fallback


@app.post("/code-interpreter")
def code_interpreter(request: CodeRequest):
    execution = execute_python_code(request.code)
    if execution["success"]:
        return {"error": [], "result": execution["output"]}

    return {
        "error": analyze_error_with_ai(request.code, execution["output"]),
        "result": execution["output"],
    }

