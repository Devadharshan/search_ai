from fastapi import FastAPI, Query
from pathlib import Path
import gzip
import re
import rapidfuzz
import time
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
import os

app = FastAPI()
LOG_DIR = "logs"  # Directory where log files are stored

# --------------------------- Load Open-LLaMA Model ---------------------------
MODEL_DIR = "C:/path_to_your_model"
MODEL_FILE = os.path.join(MODEL_DIR, "consolidated.00.pth")
TOKENIZER_FILE = os.path.join(MODEL_DIR, "tokenizer.model")

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_FILE}")
if not os.path.exists(TOKENIZER_FILE):
    raise FileNotFoundError(f"❌ Tokenizer file not found: {TOKENIZER_FILE}")

config = LlamaConfig(
    vocab_size=32000, hidden_size=4096, num_hidden_layers=32,
    num_attention_heads=32, intermediate_size=11008, max_position_embeddings=2048,
    rms_norm_eps=1e-6
)

tokenizer = LlamaTokenizer(TOKENIZER_FILE)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LlamaForCausalLM(config)
model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
model.to(device)

print("✅ Open-LLaMA Model Loaded Successfully")


# --------------------------- Log Processing Functions ---------------------------
def read_logs_from_directory(directory: str):
    """ Reads all .log and .gz files, groups multiline log entries. """
    logs = {}
    path = Path(directory)

    for file in path.rglob("*.log"):
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            logs[file.name] = group_multiline_logs(f.readlines())

    for file in path.rglob("*.gz"):
        with gzip.open(file, "rt", encoding="utf-8", errors="ignore") as f:
            logs[file.name] = group_multiline_logs(f.readlines())

    return logs


def group_multiline_logs(lines):
    """ Groups multi-line log entries into single logs. """
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


def search_logs(logs, keyword):
    """ Searches logs for a keyword using fuzzy matching. """
    results = []
    for filename, entries in logs.items():
        for entry in entries:
            if rapidfuzz.fuzz.partial_ratio(keyword.lower(), entry.lower()) > 75:
                timestamp = extract_timestamp(entry)
                service_call = extract_service_call(entry)
                ai_insight = analyze_log_with_ai(entry)

                results.append({
                    "file": filename,
                    "timestamp": timestamp,
                    "log_entry": entry,
                    "service_call": service_call,
                    "ai_insight": ai_insight
                })
    return results


def extract_timestamp(log_entry):
    """ Extracts timestamp from log entry. """
    match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', log_entry)
    return match.group(0) if match else "Unknown"


def extract_service_call(log_entry):
    """ Extracts service call if present. """
    match = re.search(r'Calling service: ([\w-]+)', log_entry)
    return match.group(1) if match else "None"


def analyze_log_with_ai(log_entry):
    """ Uses Open-LLaMA to analyze log entry. """
    input_ids = tokenizer.encode(log_entry, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(input_ids, max_length=100)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


# --------------------------- FastAPI Endpoints ---------------------------
@app.get("/search")
def search(query: str = Query(..., title="Search Keyword")):
    """ Searches logs for a keyword and returns AI analysis. """
    start_time = time.time()

    logs = read_logs_from_directory(LOG_DIR)
    results = search_logs(logs, query)

    end_time = time.time()
    search_time = round(end_time - start_time, 2)

    return {
        "query": query,
        "matches": len(results),
        "search_time_seconds": search_time,
        "results": results
    }