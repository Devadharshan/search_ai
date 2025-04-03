from fastapi import FastAPI, Query
from pathlib import Path
import gzip
import re
import rapidfuzz
from typing import List, Dict, Generator
import concurrent.futures

app = FastAPI()
LOG_DIR = "logs"  # Directory where log files are stored


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
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_log_entry, log, keyword) for log in read_logs_from_directory(LOG_DIR)]
        for future in concurrent.futures.as_completed(futures):
            if future.result():
                results.append(future.result())
    return results[:50]  # Limit to top 50 results for faster response


def process_log_entry(log: Dict[str, str], keyword: str) -> Dict:
    entry = log["log_entry"]
    if keyword.lower() in entry.lower() or rapidfuzz.fuzz.partial_ratio(keyword.lower(), entry.lower()) > 75:
        return {
            "file": log["file"],
            "timestamp": extract_timestamp(entry),
            "log_entry": entry,
            "service_call": extract_service_call(entry)
        }
    return {}


def extract_timestamp(log_entry: str) -> str:
    match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', log_entry)
    return match.group(0) if match else "Unknown"


def extract_service_call(log_entry: str) -> str:
    match = re.search(r'Calling service: ([\w-]+)', log_entry)
    return match.group(1) if match else "None"


@app.get("/search")
def search(query: str = Query(..., title="Search Keyword")):
    results = search_logs(query)
    return {"query": query, "matches": len(results), "results": results}
