import os
import json
import hashlib
import requests
import time
import threading
import psutil
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from prometheus_client import start_http_server, Gauge, Counter

# ---------- CONFIGURATION ----------
MODEL_PATH = "./path_to_your_mistral_model"
JSON_PATH = "apps_config.json"
CACHE_FILE = "scraped_cache.json"
SERVICE_NAME = "ai_webscraper"
PROMETHEUS_PORT = 8000

# ---------- PROMETHEUS METRICS ----------
cpu_gauge = Gauge('cpu_usage_percent', 'CPU usage', ['service'])
mem_gauge = Gauge('memory_usage_percent', 'Memory usage', ['service'])
perf_gauge = Gauge('ai_pertinence_score', 'AI pertinence score', ['service'])
up_counter = Counter('app_up', 'App up counter', ['service'])

def collect_metrics():
    while True:
        proc = psutil.Process(os.getpid())
        cpu = proc.cpu_percent()
        mem = proc.memory_percent()
        cpu_gauge.labels(service=SERVICE_NAME).set(cpu)
        mem_gauge.labels(service=SERVICE_NAME).set(mem)
        perf_gauge.labels(service=SERVICE_NAME).set(85.0)  # Stub score
        up_counter.labels(service=SERVICE_NAME).inc()
        time.sleep(10)

# ---------- LOAD MODEL ----------
def load_mistral(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return tokenizer, model

# ---------- SCRAPING & CACHE ----------
def scrape_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        content = soup.get_text()
        return content.strip()
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

def load_or_scrape(json_path):
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)

    with open(json_path, "r") as f:
        data = json.load(f)

    result = {}
    for app in data.get("apps", []):
        app_name = app.get("app_name", "UnknownApp")
        for func in app.get("functionalities", []):
            for url in func.get("urls", []):
                content = scrape_url(url)
                key = hashlib.sha256((app_name + url).encode()).hexdigest()
                result[key] = {
                    "app": app_name,
                    "url": url,
                    "content": content
                }

    with open(CACHE_FILE, "w") as f:
        json.dump(result, f, indent=2)

    return result

# ---------- AI SEARCH ----------
def find_relevant_context(question, scraped_data):
    keywords = question.lower().split()
    best_match = ""
    highest_score = 0

    for entry in scraped_data.values():
        content = entry["content"].lower()
        score = sum(word in content for word in keywords)
        if score > highest_score:
            highest_score = score
            best_match = entry["content"]

    return best_match[:1500] if best_match else "No relevant data found."

def generate_answer(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = model.generate(inputs['input_ids'], max_length=1024)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ---------- MAIN ----------
def main():
    print("Loading Mistral model...")
    tokenizer, model = load_mistral(MODEL_PATH)

    print("Scraping or loading cached data...")
    scraped_data = load_or_scrape(JSON_PATH)

    threading.Thread(target=collect_metrics, daemon=True).start()
    start_http_server(PROMETHEUS_PORT)

    while True:
        question = input("\nAsk something (or type 'exit'): ")
        if question.lower() == "exit":
            break

        context = find_relevant_context(question, scraped_data)
        prompt = f"Answer this using the following data:\n\n{context}\n\nQuestion: {question}\nAnswer:"
        answer = generate_answer(prompt, tokenizer, model)
        print("\n", answer.strip())

if __name__ == "__main__":
    main()