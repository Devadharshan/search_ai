import os
import gc
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from fastapi import FastAPI, Query
from pathlib import Path
import gzip
import re
import rapidfuzz
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Free up memory before loading model
gc.collect()
torch.cuda.empty_cache()

# Load GPT-2 Tokenizer and Model
MODEL_DIR = "path_to_your_gpt2_model"  # Replace with your local GPT-2 folder path
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
model.eval()
model.to("cpu")

print("GPT-2 Model Loaded Successfully!")

app = FastAPI()
LOG_DIR = "logs"  # Directory where log files are stored
cached_logs: Dict[str, List[str]] = {}  # Global log cache

@app.on_event("startup")
def load_logs():
    global cached_logs
    print("Reading logs on startup...")
    cached_logs = read_logs_from_directory(LOG_DIR)
    print(f"Loaded logs from {len(cached_logs)} files.")

def read_logs_from_directory(directory: str) -> Dict[str, List[str]]:
    logs = {}
    path = Path(directory)

    for file in path.rglob("*.log"):
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            logs[file.name] = group_multiline_logs(f.readlines())

    for file in path.rglob("*.gz"):
        with gzip.open(file, "rt", encoding="utf-8", errors="ignore") as f:
            logs[file.name] = group_multiline_logs(f.readlines())

    return logs

def group_multiline_logs(lines: List[str]) -> List[str]:
    grouped_logs = []
    temp_log = []

    for line in lines:
        if re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', line):  # New log entry
            if temp_log:
                grouped_logs.append(" ".join(temp_log))
                temp_log = []
        temp_log.append(line.strip())

    if temp_log:
        grouped_logs.append(" ".join(temp_log))

    return grouped_logs

def search_logs(logs: Dict[str, List[str]], keyword: str) -> List[Dict]:
    results = []

    def search_in_entry(entry: str, filename: str):
        if rapidfuzz.fuzz.partial_ratio(keyword.lower(), entry.lower()) > 75:
            return {
                "timestamp": extract_timestamp(entry),
                "service_call": extract_service_call(entry),
                "filename": filename
            }
        return None

    with ThreadPoolExecutor() as executor:
        futures = []
        for filename, entries in logs.items():
            for entry in entries:
                futures.append(executor.submit(search_in_entry, entry, filename))

        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    return results

def extract_timestamp(log_entry: str) -> str:
    match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', log_entry)
    return match.group(0) if match else "Unknown"

def extract_service_call(log_entry: str) -> str:
    match = re.search(r'Calling service: ([\w-]+)', log_entry)
    return match.group(1) if match else "None"

def generate_ai_insights(logs: List[str]) -> str:
    if not logs:
        return "No insights available."

    try:
        # Use top 10 frequent service calls or error lines
        lines_by_score = sorted(logs, key=lambda x: len(x), reverse=True)[:5]
        prompt = "Summarize the following service issues for a support engineer:
" + "\n".join(lines_by_score)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

        input_ids = inputs["input_ids"]
        if input_ids.size(1) == 0:
            return "AI Insight Generation Failed: Empty input after tokenization."

        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=50,
                do_sample=False
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    except RuntimeError as re:
        if "out of memory" in str(re).lower():
            return "AI Insight Generation Failed: Not enough memory."
        return f"AI Insight Generation Failed: {str(re)}"
    except Exception as e:
        return f"AI Insight Generation Failed: {str(e)}"

@app.get("/search")
def search(query: str = Query(..., title="Search Keyword"), include_ai: bool = Query(False, title="Include AI Insights")):
    results = search_logs(cached_logs, query)
    # Only extract necessary parts for AI insight
    ai_lines = [f"{r['timestamp']} - {r['service_call']} from {r['filename']}" for r in results]
    ai_insights = generate_ai_insights(ai_lines) if include_ai and results else ""
    return {"query": query, "matches": len(results), "results": results, "ai_insights": ai_insights}
