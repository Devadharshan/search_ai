import os
import gc
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
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

# Load BART Tokenizer and Model
MODEL_DIR = "path_to_your_bart_model"  # Replace with your local BART folder path
tokenizer = BartTokenizer.from_pretrained(MODEL_DIR)
model = BartForConditionalGeneration.from_pretrained(MODEL_DIR)
model.eval()
model.to("cpu")

app = FastAPI()
LOG_DIRS = ["logs", "logs_2", "logs_3"]  # Add all directories here
cached_logs: Dict[str, List[str]] = {}  # Global log cache

@app.on_event("startup")
def load_logs():
    global cached_logs
    cached_logs = read_logs_from_directories(LOG_DIRS)

def read_logs_from_directories(directories: List[str]) -> Dict[str, List[str]]:
    logs = {}
    for directory in directories:
        path = Path(directory)
        for file in path.rglob("*.log"):
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                logs[file.name] = group_multiline_logs(f.readlines())
        for file in path.rglob("*.gz"):
            with gzip.open(file, "rt", encoding="utf-8", errors="ignore") as f:
                logs[file.name] = group_multiline_logs(f.readlines())
        for file in path.rglob("*.noh"):
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                logs[file.name] = group_multiline_logs(f.readlines())
    return logs

def group_multiline_logs(lines: List[str]) -> List[str]:
    grouped_logs = []
    temp_log = []

    for line in lines:
        if re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', line):
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
                "filename": filename,
                "count": entry.lower().count(keyword.lower()),
                "log": entry
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
        log_text = "\n".join(logs)
        input_text = (
            "Summarize the following application logs with focus on user IDs, DB operations, API calls, and timing info:\n"
            + log_text[:4000]  # Truncate to prevent token overflow
        )

        inputs = tokenizer([input_text], return_tensors="pt", padding=True, truncation=True, max_length=1024)
        with torch.no_grad():
            summary_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=120,
                num_beams=4,
                early_stopping=True
            )

        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    except Exception as e:
        return f"AI Insight Generation Failed: {str(e)}"

@app.get("/search")
def search(query: str = Query(..., title="Search Keyword"), include_ai: bool = Query(False, title="Include AI Insights")):
    results = search_logs(cached_logs, query)
    ai_lines = [r["log"] for r in results if "log" in r]
    ai_insights = generate_ai_insights(ai_lines) if include_ai and ai_lines else ""
    for r in results:
        r.pop("log", None)
    return {"query": query, "matches": len(results), "results": results, "ai_insights": ai_insights}
