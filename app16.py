import os
import gc
import torch
import asyncio
import concurrent.futures
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from fastapi import FastAPI, Query
from pathlib import Path
import gzip
import re
import rapidfuzz
from typing import List, Dict

# Memory cleanup before model load
gc.collect()
torch.cuda.empty_cache()

# === Model Load ===
MODEL_DIR = "C:/path_to_your_model"
MODEL_FILE = os.path.join(MODEL_DIR, "consolidated.00.pth")

tokenizer = LlamaTokenizer.from_pretrained(MODEL_DIR)

config = LlamaConfig(
    hidden_size=4096,
    num_attention_heads=32,
    num_hidden_layers=40,
    intermediate_size=11008
)

model = LlamaForCausalLM(config)
state_dict = torch.load(MODEL_FILE, map_location="cpu")
state_dict = {k: v.to(dtype=torch.bfloat16) for k, v in state_dict.items()}

missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"🔍 Missing Keys: {missing}")
print(f"🔍 Unexpected Keys: {unexpected}")

device = "cpu"
model.to(device)
print("✅ Model Loaded Successfully!")

# === FastAPI App ===
app = FastAPI()
LOG_DIR = "logs"
executor = concurrent.futures.ThreadPoolExecutor()

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

# === AI Insight (Sync + Wrapped in Executor) ===
def generate_ai_insights_sync(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(**inputs, max_new_tokens=150, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

async def generate_ai_insights_async(prompt: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, generate_ai_insights_sync, prompt)

# === API Endpoint ===
@app.get("/search")
async def search(query: str = Query(..., title="Search Keyword")):
    logs = read_logs_from_directory(LOG_DIR)
    results = search_logs(logs, query)
    prompt = "Analyze the following logs:\n" + "\n".join([log['log_entry'] for log in results[:5]])
    ai_insights = await generate_ai_insights_async(prompt)
    return {"query": query, "matches": len(results), "results": results, "ai_insights": ai_insights}