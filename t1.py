from fastapi import FastAPI, Query
from typing import List
import os
import sys

# Add path to llama_cpp.py (adjust if needed)
sys.path.append("./llama-cpp-python")
from llama_cpp import Llama

app = FastAPI()

# List of log directories (you can add more here)
log_dirs = [
    "./logs/service1",
    "./logs/service2"
]

# Load TinyLlama model (q4 version here, you can switch to q5)
llm = Llama(model_path="./tinyllama-1.1b-chat-q4.gguf", n_ctx=2048)

# ------------------------------------------
# Keyword Search Endpoint
# ------------------------------------------
def search_keyword_in_files(directories: List[str], keyword: str):
    keyword = keyword.lower()
    total_count = 0
    matched_files = []

    for root_dir in directories:
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as file:
                        content = file.read().lower()
                        count = content.count(keyword)
                        if count > 0:
                            matched_files.append({"file": filepath, "count": count})
                            total_count += count
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")

    return {
        "total_count": total_count,
        "matched_files": matched_files
    }

@app.get("/search")
def search_logs(keyword: str = Query(..., description="Keyword to search in log files")):
    return search_keyword_in_files(log_dirs, keyword)


# ------------------------------------------
# GenAI Log Analysis Endpoint
# ------------------------------------------
@app.get("/genai")
def genai_log_analysis(log: str = Query(..., description="Paste log line or error message")):
    prompt = f"""You are a helpful assistant for production support. Analyze the following log and provide the most likely root cause and suggested fix.

Log:
{log}

Response:"""
    
    response = llm(prompt, max_tokens=256, stop=["</s>"])
    return {
        "log": log,
        "analysis": response["choices"][0]["text"].strip()
    }