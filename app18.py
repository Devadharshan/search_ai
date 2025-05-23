import os
import gc
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from fastapi import FastAPI, Query
from pathlib import Path
import gzip
import re
import rapidfuzz
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Free up memory before loading model
gc.collect()
torch.cuda.empty_cache()

# Load GPT-2 Tokenizer and Model
MODEL_DIR = "path_to_your_gpt2_model"  # Replace with your local GPT-2 folder path
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
model.eval()
model.to("cpu")

print("GPT-2 Model Loaded Successfully!")

app = FastAPI()
LOG_DIR = "logs"  # Directory where log files are stored
cached_logs: Dict[str, List[str]] = {}  # Global log cache

@app.on_event("startup")
def load_logs():
    global cached_logs
    print("Reading logs on startup...")
    cached_logs = read_logs_from_directory(LOG_DIR)
    print(f"Loaded logs from {len(cached_logs)} files.")

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

def search_logs(logs: Dict[str, List[str]], keyword: str) -> List[str]:
    results = []

    def search_in_entry(entry: str):
        if rapidfuzz.fuzz.partial_ratio(keyword.lower(), entry.lower()) > 75:
            return entry
        return None

    with ThreadPoolExecutor() as executor:
        futures = []
        for entries in logs.values():
            for entry in entries:
                futures.append(executor.submit(search_in_entry, entry))

        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    return results

def generate_ai_insights(entries: List[str]) -> str:
    if not entries:
        return "No insights available."

    try:
        text_chunk = "\n".join(entries)
        prompt = f"Analyze the following logs and provide insights and issues:\n{text_chunk}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)
        input_ids = inputs["input_ids"]

        if torch.any(input_ids >= model.config.vocab_size):
            return "AI Insight Generation Failed: input index exceeds vocabulary size."

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=150,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    except Exception as e:
        return f"AI Insight Generation Failed: {str(e)}"

@app.get("/search")
def search(query: str = Query(..., title="Search Keyword"), include_ai: bool = Query(False, title="Include AI Insights")):
    matches = search_logs(cached_logs, query)
    ai_insights = generate_ai_insights(matches) if include_ai and matches else ""
    return {"query": query, "matches": len(matches), "ai_insights": ai_insights}
