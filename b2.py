import os
import gc
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from fastapi import FastAPI, Query
from pathlib import Path
import gzip
import re
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Memory cleanup
gc.collect()
torch.cuda.empty_cache()

# Load BART tokenizer and model
MODEL_DIR = "path_to_your_bart_model"  # Replace with your local BART folder path
tokenizer = BartTokenizer.from_pretrained(MODEL_DIR)
model = BartForConditionalGeneration.from_pretrained(MODEL_DIR)
model.eval()
model.to("cpu")

app = FastAPI()
LOG_DIRS = ["logs1", "logs2", "logs3"]  # Add multiple directories here
cached_logs: Dict[str, List[str]] = {}

@app.on_event("startup")
def load_logs():
    global cached_logs
    for dir_path in LOG_DIRS:
        logs = read_logs_from_directory(dir_path)
        cached_logs.update(logs)

def read_logs_from_directory(directory: str) -> Dict[str, List[str]]:
    logs = {}
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
        if keyword.lower() in entry.lower():
            return {
                "timestamp": extract_timestamp(entry),
                "service_call": extract_service_call(entry),
                "filename": filename,
                "count": entry.lower().count(keyword.lower())
            }
        return None
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(search_in_entry, entry, filename)
                   for filename, entries in logs.items() for entry in entries]
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

def preprocess_logs_for_ai(logs: List[str]) -> List[str]:
    return [line for line in logs if any(keyword in line.lower() for keyword in ["user", "db", "sql", "api", "error", "request", "response"])]

def chunk_logs(log_lines: List[str], max_tokens: int = 1024) -> List[str]:
    chunks = []
    current_chunk = []
    total_tokens = 0
    for line in log_lines:
        tokens = tokenizer.tokenize(line)
        if total_tokens + len(tokens) > max_tokens:
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            total_tokens = len(tokens)
        else:
            current_chunk.append(line)
            total_tokens += len(tokens)
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    return chunks

def generate_ai_insights(logs: List[str]) -> str:
    try:
        important_lines = preprocess_logs_for_ai(logs)
        if not important_lines:
            return "No significant log lines for insights."
        chunks = chunk_logs(important_lines, max_tokens=1024)
        insights = []
        for chunk in chunks:
            input_ids = tokenizer(chunk, return_tensors="pt", max_length=1024, truncation=True).input_ids
            summary_ids = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            insights.append(summary)
        return "\n---\n".join(insights)
    except Exception as e:
        return f"AI Insight Generation Failed: {str(e)}"

@app.get("/search")
def search(query: str = Query(..., title="Search Keyword"), include_ai: bool = Query(False, title="Include AI Insights")):
    results = search_logs(cached_logs, query)
    ai_lines = [entry for file in cached_logs.values() for entry in file if query.lower() in entry.lower()]
    ai_insights = generate_ai_insights(ai_lines) if include_ai else ""
    return {"query": query, "matches": len(results), "results": results, "ai_insights": ai_insights}
