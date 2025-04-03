import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from fastapi import FastAPI, Query
from pathlib import Path
import gzip
import re
import time
import concurrent.futures
import rapidfuzz
from typing import List, Dict

app = FastAPI()
LOG_DIR = "logs"  # Directory where log files are stored

# Load Open-LLaMA-13B Model
MODEL_PATH = "C:/path/to/Open-LLaMA-13B/consolidated.00.pth"
TOKENIZER_PATH = "C:/path/to/Open-LLaMA-13B/tokenizer.model"

print("Loading Open-LLaMA-13B model...")
tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)
model = LlamaForCausalLM.from_pretrained("openlm-research/open_llama_13b")
state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model.eval()
print("Model loaded successfully!")


def read_logs_from_directory(directory: str) -> Dict[str, List[str]]:
    """Reads logs from a directory including both normal and gzipped logs."""
    logs = {}
    path = Path(directory)

    def read_file(file):
        """Reads and processes a single log file."""
        if file.suffix == ".gz":
            with gzip.open(file, "rt", encoding="utf-8", errors="ignore") as f:
                return group_multiline_logs(f.readlines())
        else:
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                return group_multiline_logs(f.readlines())

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(read_file, file): file for file in path.rglob("*")}
        for future in concurrent.futures.as_completed(future_to_file):
            file = future_to_file[future]
            try:
                logs[file.name] = future.result()
            except Exception as e:
                print(f"Error reading {file}: {e}")

    return logs


def group_multiline_logs(lines: List[str]) -> List[str]:
    """Groups multi-line log entries into single entries."""
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
    """Performs fuzzy search on logs and extracts relevant information."""
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
    """Extracts timestamp from a log entry."""
    match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', log_entry)
    return match.group(0) if match else "Unknown"


def extract_service_call(log_entry: str) -> str:
    """Extracts service call details from a log entry."""
    match = re.search(r'Calling service: ([\w-]+)', log_entry)
    return match.group(1) if match else "None"


def generate_ai_summary(log_entries: List[str]) -> str:
    """Uses Open-LLaMA-13B to summarize the extracted logs."""
    if not log_entries:
        return "No relevant log data found."

    input_text = "Summarize the following logs:\n" + "\n".join(log_entries[:5])  # Limit to 5 logs for processing
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=150, num_return_sequences=1)

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


@app.get("/search")
def search(query: str = Query(..., title="Search Keyword")):
    """Search logs for a given keyword and return results with AI insights."""
    start_time = time.time()
    
    logs = read_logs_from_directory(LOG_DIR)
    results = search_logs(logs, query)
    search_time = round(time.time() - start_time, 2)

    # AI-based summary of found logs
    ai_summary = generate_ai_summary([r["log_entry"] for r in results])

    return {
        "query": query,
        "matches": len(results),
        "search_time_seconds": search_time,
        "results": results,
        "ai_summary": ai_summary
    }