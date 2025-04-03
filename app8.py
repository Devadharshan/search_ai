from fastapi import FastAPI, Query
from pathlib import Path
import gzip
import re
import asyncio
import rapidfuzz
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import time

app = FastAPI()
LOG_DIR = "logs"  # Log file directory

# Load Open-LLaMA from local
MODEL_PATH = "C:/path_to_your_model/consolidated.00.pth"
TOKENIZER_PATH = "C:/path_to_your_model/tokenizer.model"

# Load AI Model & Tokenizer
tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)
model = LlamaForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")

# --------------------------- Log Processing Functions ---------------------------

def read_logs_from_directory(directory: str):
    logs = {}
    path = Path(directory)
    
    for file in path.rglob("*.log"):
        logs[file.name] = process_file(file)
    
    for file in path.rglob("*.gz"):
        logs[file.name] = process_gz_file(file)

    return logs


def process_file(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return list(f)  # Read line by line for efficiency


def process_gz_file(file_path):
    with gzip.open(file_path, "rt", encoding="utf-8", errors="ignore") as f:
        return list(f)


async def search_logs_async(logs, keyword):
    results = []
    tasks = [asyncio.create_task(search_file_async(filename, entries, keyword)) for filename, entries in logs.items()]
    found_entries = await asyncio.gather(*tasks)
    
    for entry_list in found_entries:
        results.extend(entry_list)
    
    return results


async def search_file_async(filename, entries, keyword):
    matched_entries = []
    for entry in entries:
        if keyword.lower() in entry.lower():  # Faster search
            timestamp = extract_timestamp(entry)
            service_call = extract_service_call(entry)
            matched_entries.append({
                "file": filename,
                "timestamp": timestamp,
                "log_entry": entry.strip(),
                "service_call": service_call
            })
    return matched_entries


def extract_timestamp(log_entry):
    match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', log_entry)
    return match.group(0) if match else "Unknown"


def extract_service_call(log_entry):
    match = re.search(r'Calling service: ([\w-]+)', log_entry)
    return match.group(1) if match else "None"

# --------------------------- AI Analysis ---------------------------

def analyze_logs_with_ai(logs):
    if not logs:
        return "No relevant logs found."
    
    prompt = "Analyze these logs and find anomalies:\n" + "\n".join(logs[:20])  # Limit to 20 logs
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=500)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# --------------------------- FastAPI Endpoints ---------------------------

@app.get("/search")
async def search(query: str = Query(..., title="Search Keyword")):
    start_time = time.time()

    logs = read_logs_from_directory(LOG_DIR)
    results = await search_logs_async(logs, query)

    ai_analysis = analyze_logs_with_ai([r["log_entry"] for r in results])
    
    execution_time = round(time.time() - start_time, 2)
    
    return {
        "query": query,
        "matches": len(results),
        "execution_time": f"{execution_time} seconds",
        "results": results,
        "ai_analysis": ai_analysis
    }