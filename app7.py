from fastapi import FastAPI, Query
from pathlib import Path
import gzip
import re
import time
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from typing import List, Dict
import rapidfuzz

app = FastAPI()
LOG_DIR = "logs"  # Change this to your log file directory

# Paths for locally stored Open-LLaMA-13B
MODEL_PATH = r"C:\path\to\Open-LLaMA-13B\consolidated.00.pth"
TOKENIZER_PATH = r"C:\path\to\Open-LLaMA-13B\tokenizer.model"

# Load LLaMA Model
tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)
model = LlamaForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32, device_map="auto")
model.eval()


def read_logs_from_directory(directory: str) -> Dict[str, List[str]]:
    """ Reads multiple log files (both .log and .gz) and returns a dictionary of logs. """
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
    """ Groups log entries that span multiple lines. """
    grouped_logs = []
    temp_log = []

    for line in lines:
        if re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', line):  # Detect new log entry
            if temp_log:
                grouped_logs.append(" ".join(temp_log))
                temp_log = []
        temp_log.append(line.strip())

    if temp_log:
        grouped_logs.append(" ".join(temp_log))

    return grouped_logs


def search_logs(logs: Dict[str, List[str]], keyword: str) -> List[Dict]:
    """ Performs fuzzy search and extracts metadata (timestamp, service calls). """
    results = []
    for filename, entries in logs.items():
        for entry in entries:
            if rapidfuzz.fuzz.partial_ratio(keyword.lower(), entry.lower()) > 75:
                timestamp = extract_timestamp(entry)
                service_call = extract_service_call(entry)
                results.append({
                    "file": filename,
                    "timestamp": timestamp,
                    "log_entry": highlight_keyword(entry, keyword),
                    "service_call": service_call
                })
    return results


def extract_timestamp(log_entry: str) -> str:
    """ Extracts the timestamp from a log entry. """
    match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', log_entry)
    return match.group(0) if match else "Unknown"


def extract_service_call(log_entry: str) -> str:
    """ Extracts service calls if another service is being called. """
    match = re.search(r'Calling service: ([\w-]+)', log_entry)
    return match.group(1) if match else "None"


def highlight_keyword(log_entry: str, keyword: str) -> str:
    """ Highlights the keyword in the log entry. """
    return re.sub(f"({re.escape(keyword)})", r"**\1**", log_entry, flags=re.IGNORECASE)


def generate_ai_insight(query: str, log_data: str) -> str:
    """ Uses Open-LLaMA-13B to analyze logs and provide AI insights. """
    prompt = f"Analyze the following log data and provide insights:\n\n{log_data}\n\nInsights:"
    input_tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cpu")
    output = model.generate(**input_tokens, max_new_tokens=200)
    return tokenizer.decode(output[0], skip_special_tokens=True)


@app.get("/search")
def search(query: str = Query(..., title="Search Keyword")):
    """ API endpoint for searching logs and getting AI-based insights. """
    start_time = time.time()
    logs = read_logs_from_directory(LOG_DIR)
    results = search_logs(logs, query)

    # Combine log entries for AI analysis (if results exist)
    log_text = "\n".join([res["log_entry"] for res in results[:5]])  # Limit logs for AI processing
    ai_insight = generate_ai_insight(query, log_text) if log_text else "No AI insights available."

    elapsed_time = time.time() - start_time
    return {
        "query": query,
        "matches": len(results),
        "search_time_seconds": round(elapsed_time, 2),
        "ai_insight": ai_insight,
        "results": results
    }