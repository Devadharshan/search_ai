import os
import json
import requests
import hashlib
import logging
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from bs4 import BeautifulSoup
import time
import subprocess

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
BASE_DIR = "/path/to/your/gpt2/model/files"  # <-- Change this
JSON_FILE_PATH = "apps_data.json"  # Your input JSON
OPTIMIZED_JSON_PATH = "optimized_apps_data.json"
CACHE_DIR = "cache"

# Load GPT-2
tokenizer = GPT2Tokenizer.from_pretrained(BASE_DIR)
model = GPT2LMHeadModel.from_pretrained(BASE_DIR)
model.eval()

# Web Scraping with Session and Cache
session = requests.Session()
session.verify = "/path/to/certificate.pem"  # <-- Change if needed

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def fetch_and_cache(url):
    url_hash = hashlib.md5(url.encode()).hexdigest()
    cache_file = os.path.join(CACHE_DIR, f"{url_hash}.html")

    if os.path.exists(cache_file):
        logging.info(f"Using cached version of {url}")
        with open(cache_file, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        logging.info(f"Fetching {url}")
        try:
            response = session.get(url, timeout=10)
            response.raise_for_status()
            content = response.text
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            logging.error(f"Failed to fetch {url}: {e}")
            content = ""
    return content

def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove scripts, styles
    for tag in soup(["script", "style"]):
        tag.decompose()

    # Extract all tables and images
    tables = [str(table) for table in soup.find_all("table")]
    images = [img.get("src") for img in soup.find_all("img") if img.get("src")]

    # Remaining text
    text = soup.get_text(separator="\n", strip=True)

    return {
        "text": text,
        "tables": tables,
        "images": images
    }

# Process JSON and fetch web content
def process_json_file():
    with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    optimized_data = []

    for app in data:
        app_info = {
            "app_name": app["app_name"],
            "about": app["about"],
            "urls_data": []
        }
        for url in app.get("urls", []):
            html_content = fetch_and_cache(url)
            extracted = extract_text_from_html(html_content)
            app_info["urls_data"].append({
                "url": url,
                "content": extracted["text"],
                "tables": extracted["tables"],
                "images": extracted["images"]
            })
        optimized_data.append(app_info)

    # Save optimized JSON
    with open(OPTIMIZED_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(optimized_data, f, indent=4)

    return optimized_data

# Search JSON content for matching info
def search_in_json(user_query, optimized_data):
    collected_context = ""
    for app in optimized_data:
        if user_query.lower() in app['app_name'].lower() or user_query.lower() in app['about'].lower():
            collected_context += f"\nAbout {app['app_name']}:\n{app['about']}\n"
            for url_data in app['urls_data']:
                collected_context += f"\nFrom {url_data['url']}:\n{url_data['content'][:500]}...\n"

    if not collected_context.strip():
        collected_context = "No relevant information found in the knowledge base."

    # Now truncate the context if too long
    tokenized_context = tokenizer(collected_context, return_tensors="pt", truncation=True, max_length=600)
    decoded_context = tokenizer.decode(tokenized_context['input_ids'][0], skip_special_tokens=True)

    return decoded_context

# Generate response using local GPT-2
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=900)
    attention_mask = inputs.get("attention_mask")

    output = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=attention_mask,
        max_length=1024,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded

# Main interactive loop
def main():
    logging.info("Starting Knowledge Base Processor...")

    # Step 1: Process JSON and scrape/caching
    optimized_data = process_json_file()

    logging.info("Ready to answer your queries using local GPT-2 model!")

    # Step 2: Q&A Loop
    while True:
        user_input = input("\nAsk your question (or type 'exit'): ").strip()
        if user_input.lower() == "exit":
            break

        try:
            context = search_in_json(user_input, optimized_data)
            prompt = f"Based on the following knowledge, answer the query:\n{context}\nQuery: {user_input}\nAnswer:"
            answer = generate_response(prompt)
            print("\nAnswer:", answer)
        except Exception as e:
            logging.error(f"Error generating answer: {e}")

if __name__ == "__main__":
    main()