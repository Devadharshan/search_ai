import os
import gc
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from fastapi import FastAPI, Query
from pathlib import Path
import gzip
import re
import rapidfuzz
from typing import List, Dict

# ✅ Free up memory before loading model
gc.collect()
torch.cuda.empty_cache()

# ✅ Define Paths
MODEL_DIR = "C:/path_to_your_model"
MODEL_FILE = os.path.join(MODEL_DIR, "consolidated.00.pth")
TOKENIZER_FILE = os.path.join(MODEL_DIR, "tokenizer.model")

# ✅ Load Tokenizer (Lazy Load to Reduce Memory Usage)
tokenizer = LlamaTokenizer.from_pretrained(MODEL_DIR)

# ✅ Load Model Configuration (Optimized for CPU)
config = LlamaConfig(
    hidden_size=4096, 
    num_attention_heads=32, 
    num_hidden_layers=40, 
    intermediate_size=11008
)

# ✅ Initialize Model with Optimized Memory Settings
model = LlamaForCausalLM(config)

# ✅ Load State Dict Efficiently (Convert to bfloat16 to Reduce Memory)
state_dict = torch.load(MODEL_FILE, map_location="cpu")
state_dict = {k: v.to(dtype=torch.bfloat16) for k, v in state_dict.items()}  # ⚡ Optimized for CPU

# ✅ Handle Missing Keys
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"🔍 Missing Keys: {missing}")
print(f"🔍 Unexpected Keys: {unexpected}")

# ✅ Move Model to CPU with Controlled Memory Usage
device = "cpu"
model.to(device)

print("✅ Model Loaded Successfully!")

# =========================
# 🚀 FastAPI Log Search Backend
# =========================
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

def generate_ai_insights(logs: List[Dict]) -> str:
    if not logs:
        return "No insights available."
    
    prompt = "Analyze the following logs and summarize key insights and potential issues:\n" + "\n".join([log['log_entry'] for log in logs[:5]])
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(**inputs, max_new_tokens=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.get("/search")
def search(query: str = Query(..., title="Search Keyword")):
    logs = read_logs_from_directory(LOG_DIR)
    results = search_logs(logs, query)
    ai_insights = generate_ai_insights(results)
    return {"query": query, "matches": len(results), "results": results, "ai_insights": ai_insights}
