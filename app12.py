from fastapi import FastAPI, Query
from pathlib import Path
import gzip
import re
import rapidfuzz
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import time
from typing import List, Dict

app = FastAPI()
LOG_DIR = "logs"  # Change this to your actual log directory

# Load AI Model
MODEL_PATH = "C:/path_to_your_model/"  # Update with your actual model directory
TOKENIZER_PATH = f"{MODEL_PATH}/tokenizer.model"
MODEL_FILE = f"{MODEL_PATH}/consolidated.00.pth"

# Load Tokenizer & Model
tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)
model = LlamaForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))

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

def search_logs(logs: Dict[str, List[str]], keyword: str) -> Dict[str, List[Dict]]:
    results = {}
    total_occurrences = 0
    for filename, entries in logs.items():
        file_results = []
        file_occurrences = 0
        for entry in entries:
            match_count = entry.lower().count(keyword.lower())
            if match_count > 0:
                timestamp = extract_timestamp(entry)
                service_call = extract_service_call(entry)
                file_results.append({
                    "file": filename,
                    "timestamp": timestamp,
                    "log_entry": entry,
                    "service_call": service_call,
                    "occurrences": match_count
                })
                file_occurrences += match_count
        
        if file_results:
            results[filename] = file_results
            total_occurrences += file_occurrences
    
    return {"total_occurrences": total_occurrences, "results": results}

def extract_timestamp(log_entry: str) -> str:
    match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', log_entry)
    return match.group(0) if match else "Unknown"

def extract_service_call(log_entry: str) -> str:
    match = re.search(r'Calling service: ([\w-]+)', log_entry)
    return match.group(1) if match else "None"

def generate_ai_summary(keyword: str, logs: Dict) -> str:
    """ AI analyzes logs and generates a summary """
    log_text = "\n".join([" ".join(item['log_entry']) for key, value in logs['results'].items() for item in value])
    input_text = f"Analyze the following logs for keyword '{keyword}': {log_text[:1000]}..."  # Limit input length
    
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048).to("cpu")
    with torch.no_grad():
        output = model.generate(**inputs, max_length=512)
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

@app.get("/search")
def search(query: str = Query(..., title="Search Keyword")):
    start_time = time.time()
    logs = read_logs_from_directory(LOG_DIR)
    search_results = search_logs(logs, query)
    
    ai_summary = generate_ai_summary(query, search_results)
    end_time = time.time()
    
    return {
        "query": query,
        "total_occurrences": search_results["total_occurrences"],
        "search_time_seconds": round(end_time - start_time, 2),
        "results": search_results["results"],
        "ai_analysis": ai_summary
    }
