from fastapi import FastAPI, Query
from pathlib import Path
import gzip
import re
import rapidfuzz
from typing import List, Dict, Generator
import concurrent.futures
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()
LOG_DIR = "logs"  # Directory where log files are stored
MODEL_PATH = "C:/path/to/Open-LLaMA-13B/consolidated.00.pth"

# Load Open-LLaMA model
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def read_logs_from_directory(directory: str) -> Generator[Dict[str, str], None, None]:
    path = Path(directory)
    for file in path.rglob("*.log"):
        yield from read_log_file(file)
    for file in path.rglob("*.gz"):
        yield from read_gz_file(file)


def read_log_file(file: Path) -> Generator[Dict[str, str], None, None]:
    with open(file, "r", encoding="utf-8", errors="ignore") as f:
        for log_entry in group_multiline_logs(f):
            yield {"file": file.name, "log_entry": log_entry}


def read_gz_file(file: Path) -> Generator[Dict[str, str], None, None]:
    with gzip.open(file, "rt", encoding="utf-8", errors="ignore") as f:
        for log_entry in group_multiline_logs(f):
            yield {"file": file.name, "log_entry": log_entry}


def group_multiline_logs(lines) -> Generator[str, None, None]:
    temp_log = []
    for line in lines:
        if re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', line):  # New log entry
            if temp_log:
                yield " ".join(temp_log)
                temp_log = []
        temp_log.append(line.strip())
    if temp_log:
        yield " ".join(temp_log)


def search_logs(keyword: str) -> List[Dict]:
    results = []
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_log_entry, log, keyword) for log in read_logs_from_directory(LOG_DIR)]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                results.extend(result)  # Append all matches instead of just one
    end_time = time.time()
    search_time = round(end_time - start_time, 2)
    return results, search_time


def process_log_entry(log: Dict[str, str], keyword: str) -> List[Dict]:
    matches = []
    entry = log["log_entry"]
    if keyword.lower() in entry.lower() or rapidfuzz.fuzz.partial_ratio(keyword.lower(), entry.lower()) > 75:
        matches.append({
            "file": log["file"],
            "timestamp": extract_timestamp(entry),
            "log_entry": entry,
            "service_call": extract_service_call(entry)
        })
    return matches


def extract_timestamp(log_entry: str) -> str:
    match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', log_entry)
    return match.group(0) if match else "Unknown"


def extract_service_call(log_entry: str) -> str:
    match = re.search(r'Calling service: ([\w-]+)', log_entry)
    return match.group(1) if match else "None"


def generate_ai_summary(logs: List[Dict]) -> str:
    prompt = "Summarize the following logs and identify any potential issues:\n" + "\n".join(log["log_entry"] for log in logs[:10])  # Limit AI input size
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(**inputs, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


@app.get("/search")
def search(query: str = Query(..., title="Search Keyword")):
    results, search_time = search_logs(query)
    ai_summary = generate_ai_summary(results) if results else "No relevant logs found."
    return {"query": query, "matches": len(results), "search_time": f"{search_time} seconds", "results": results, "ai_summary": ai_summary}
