import os
import gc
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
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

# Load BART Tokenizer and Model
MODEL_DIR = "path_to_your_local_bart_model"  # Update this with actual path
tokenizer = BartTokenizer.from_pretrained(MODEL_DIR)
model = BartForConditionalGeneration.from_pretrained(MODEL_DIR)
model.eval()
model.to("cpu")

app = FastAPI()
LOG_DIRS = ["logs", "logs_service1", "logs_service2"]  # Multiple directories
cached_logs: Dict[str, List[str]] = {}  # Global log cache

@app.on_event("startup")
def load_logs():
    global cached_logs
    cached_logs = {}
    for directory in LOG_DIRS:
        cached_logs.update(read_logs_from_directory(directory))

def read_logs_from_directory(directory: str) -> Dict[str, List[str]]:
    logs = {}
    path = Path(directory)

    for ext in ("*.log", "*.gz", "*.noh"):
        for file in path.rglob(ext):
            try:
                if file.suffix == ".gz":
                    with gzip.open(file, "rt", encoding="utf-8", errors="ignore") as f:
                        logs[file.name] = group_multiline_logs(f.readlines())
                else:
                    with open(file, "r", encoding="utf-8", errors="ignore") as f:
                        logs[file.name] = group_multiline_logs(f.readlines())
            except Exception:
                continue

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

    def search_in_entry(entry: str, filename: str):
        if rapidfuzz.fuzz.partial_ratio(keyword.lower(), entry.lower()) > 75:
            highlighted = highlight_keyword(entry, keyword)
            return {
                "timestamp": extract_timestamp(entry),
                "service_call": extract_service_call(entry),
                "filename": filename,
                "count": entry.lower().count(keyword.lower()),
                "log": highlighted
            }
        return None

    with ThreadPoolExecutor() as executor:
        futures = []
        for filename, entries in logs.items():
            for entry in entries:
                futures.append(executor.submit(search_in_entry, entry, filename))

        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    return results

def extract_timestamp(log_entry: str) -> str:
    match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', log_entry)
    return match.group(0) if match else "Unknown"

def extract_service_call(log_entry: str) -> str:
    match = re.search(r'Calling service: ([\w-]+)', log_entry)
    return match.group(1) if match else "None"

def highlight_keyword(text: str, keyword: str) -> str:
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    return pattern.sub(lambda m: f"**{m.group(0)}**", text)

def clean_log_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r'\s+', ' ', line)
    return line

def format_logs_for_bart(logs: List[str]) -> str:
    cleaned = [clean_log_line(l) for l in logs if len(l.strip()) > 10]
    deduped = list(dict.fromkeys(cleaned))
    top_logs = deduped[:50]  # limit to top 50
    return "\n".join(top_logs)

def generate_ai_insights(logs: List[str]) -> str:
    if not logs:
        return "No insights available."

    try:
        input_text = format_logs_for_bart(logs)
        prompt = (
            "Summarize the following application logs and highlight:\n"
            "- Any user IDs involved\n"
            "- Database queries or connections\n"
            "- External API calls with timings\n"
            "- Errors or failures with timestamps\n\n"
            "Logs:\n" + input_text
        )

        inputs = tokenizer([prompt], max_length=1024, truncation=True, return_tensors="pt")
        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=4,
            length_penalty=2.0,
            max_length=200,
            early_stopping=True
        )
        output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return output.strip() if output else "No relevant summary generated."

    except Exception as e:
        return f"AI Insight Generation Failed: {str(e)}"

@app.get("/search")
def search(query: str = Query(..., title="Search Keyword"), include_ai: bool = Query(False, title="Include AI Insights")):
    results = search_logs(cached_logs, query)
    ai_lines = [r["log"] for r in results if "log" in r]
    ai_insights = generate_ai_insights(ai_lines) if include_ai and ai_lines else ""
    for r in results:
        r.pop("log", None)  # Remove full log from response
    return {"query": query, "matches": len(results), "results": results, "ai_insights": ai_insights}
