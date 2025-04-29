import os
import json
import hashlib
import logging
import requests
import torch
import psutil
import time
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from prometheus_client import CollectorRegistry, Counter, Gauge, push_to_gateway

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
MODEL_PATH = "/path/to/mistral/model"
MAIN_JSON_PATH = "kb_main.json"
CACHE_DIR = "cache"
OPTIMIZED_JSON_PATH = "kb_optimized.json"
OTEL_COLLECTOR_URL = "http://otel-collector-host:4318/metrics"

# Metrics Setup
registry = CollectorRegistry()
query_counter = Counter('ai_queries_total', 'Total AI queries', ['service'], registry=registry)
cpu_gauge = Gauge('ai_cpu_usage_percent', 'CPU usage percent', ['service'], registry=registry)
mem_gauge = Gauge('ai_memory_usage_bytes', 'Memory usage in bytes', ['service'], registry=registry)
pertinence_gauge = Gauge('ai_pertinence_score', 'Pertinence of AI responses', ['service'], registry=registry)

SERVICE_NAME = "ai_kb_assistant"

# Create cache dir
os.makedirs(CACHE_DIR, exist_ok=True)

# Load Model
logging.info("Loading model from local path...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)

# Load and Cache Web Data
def hash_url(url): return hashlib.md5(url.encode()).hexdigest()

def fetch_content(url, session):
    cache_path = os.path.join(CACHE_DIR, hash_url(url) + ".txt")
    try:
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return f.read()

        response = session.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")

        text = soup.get_text(separator="\n", strip=True)
        images = [img['src'] for img in soup.find_all('img', src=True)]
        tables = [str(table) for table in soup.find_all('table')]

        content = f"TEXT:\n{text}\n\nIMAGES:\n{images}\n\nTABLES:\n{tables}"

        with open(cache_path, "w") as f:
            f.write(content)

        return content
    except Exception as e:
        logging.error(f"Error fetching {url}: {e}")
        return ""

def build_optimized_json():
    logging.info("Generating optimized JSON from web...")
    session = requests.session()
    session.verify = "/path/to/cert.pem"  # Update if needed

    with open(MAIN_JSON_PATH, "r") as f:
        data = json.load(f)

    optimized = {}
    for app in data:
        app_name = app["app_name"]
        urls = app["urls"]
        content_list = [fetch_content(url, session) for url in urls]
        optimized[app_name] = {
            "about": app["about"],
            "content": "\n\n".join(content_list)
        }

    with open(OPTIMIZED_JSON_PATH, "w") as f:
        json.dump(optimized, f, indent=2)

    logging.info("Optimized JSON saved.")
    return optimized

# Load Optimized JSON
def load_optimized_json():
    if not os.path.exists(OPTIMIZED_JSON_PATH):
        return build_optimized_json()
    with open(OPTIMIZED_JSON_PATH) as f:
        return json.load(f)

kb_data = load_optimized_json()

# AI Response
def generate_answer(question, kb):
    prompt = f"Answer this using only the knowledge base:\n\n{question}\n\n"
    for app, info in kb.items():
        prompt += f"\n[{app}]\n{info['about']}\n{info['content']}\n"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.split(question)[-1].strip()

# Monitor & Metrics
def record_metrics(service_name, pertinence_score=0.8):
    query_counter.labels(service=service_name).inc()
    cpu_gauge.labels(service=service_name).set(psutil.cpu_percent())
    mem_gauge.labels(service=service_name).set(psutil.Process(os.getpid()).memory_info().rss)
    pertinence_gauge.labels(service=service_name).set(pertinence_score)

    push_to_gateway(OTEL_COLLECTOR_URL, job=service_name, registry=registry)

# Main Function
def main():
    while True:
        question = input("Ask something (or 'exit'): ").strip()
        if question.lower() == "exit":
            break
        answer = generate_answer(question, kb_data)
        print("\nAnswer:", answer)
        record_metrics(SERVICE_NAME)

if __name__ == "__main__":
    main()