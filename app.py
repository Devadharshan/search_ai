from fastapi import FastAPI, Query
from pathlib import Path
import gzip
import re
import rapidfuzz
from typing import List, Dict

app = FastAPI()
LOG_DIR = "logs"  # Directory where log files are stored


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
    for filename, entries in logs.items():
        for entry in entries:
            if rapidfuzz.fuzz.partial_ratio(keyword.lower(), entry.lower()) > 75:
                timestamp = extract_timestamp(entry)
                service_call = extract_service_call(entry)
                results.append({
                    "file": filename,
                    "timestamp": timestamp,
                    "log_entry": entry,
                    "service_call": service_call
                })
    return results


def extract_timestamp(log_entry: str) -> str:
    match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', log_entry)
    return match.group(0) if match else "Unknown"


def extract_service_call(log_entry: str) -> str:
    match = re.search(r'Calling service: ([\w-]+)', log_entry)
    return match.group(1) if match else "None"


@app.get("/search")
def search(query: str = Query(..., title="Search Keyword")):
    logs = read_logs_from_directory(LOG_DIR)
    results = search_logs(logs, query)
    return {"query": query, "matches": len(results), "results": results}
