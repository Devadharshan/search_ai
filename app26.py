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

app = FastAPI()
LOG_DIRS = ["logs", "logs_2", "logs_3"]  # Add all your log directories here
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

    for file in path.rglob("*.log"):
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            logs[file.name] = group_multiline_logs(f.readlines())

    for file in path.rglob("*.gz"):
        with gzip.open(file, "rt", encoding="utf-8", errors="ignore") as f:
            logs[file.name] = group_multiline_logs(f.readlines())

    for file in path.rglob("*.noh"):
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
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

def generate_ai_insights(logs: List[str]) -> str:
    if not logs:
        return "No insights available."

    try:
        # Clean logs: remove non-printable lines
        clean_logs = [line for line in logs if all(32 <= ord(char) <= 126 or char in "\n\t\r" for char in line)]
        if not clean_logs:
            return "AI Insight Generation Failed: Logs too noisy or binary."

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        static_prompt = (
            "From the following logs, extract and summarize:\n"
            "- User IDs (if any)\n"
            "- DB connections or queries\n"
            "- External API calls with URLs\n"
            "- Request timings if available\n"
            "LOGS:\n"
        )

        prompt_ids = tokenizer.encode(static_prompt, add_special_tokens=False)
        log_ids = tokenizer.encode("\n".join(clean_logs), add_special_tokens=False)

        max_model_tokens = model.config.n_positions if hasattr(model.config, "n_positions") else 1024
        max_generate_tokens = 100
        allowed_input_tokens = max_model_tokens - max_generate_tokens

        total_input_ids = (prompt_ids + log_ids)[-allowed_input_tokens:]
        input_tensor = torch.tensor([total_input_ids]).to(model.device)
        attention_mask = torch.ones_like(input_tensor)

        with torch.no_grad():
            output = model.generate(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                max_new_tokens=max_generate_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False
            )

        generated = output[0][input_tensor.shape[1]:]
        decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()

        if not decoded or all(char in "0123456789ABCDEF: " for char in decoded):
            return "AI Insight Generation Failed: Output looks malformed. Consider reducing input noise."

        return decoded

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
