import os
import json
import hashlib
import logging
import requests
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from bs4 import BeautifulSoup
from functools import lru_cache

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
CACHE_DIR = "./cache"
MAIN_JSON = "./apps_data.json"  # <<<<<< MAIN JSON TO READ (not optimized one)
OPTIMIZED_JSON = "./optimized_apps_data.json"
MODEL_DIR = "./gpt2_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Initialize requests session
session = requests.Session()
# session.auth = HTTPKerberosAuth()  # Uncomment if needed
# session.verify = '/path/to/cert.pem'  # Uncomment if needed


# Function to fetch and scrape URL content with caching
def fetch_data(url):
    try:
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_file = os.path.join(CACHE_DIR, f"{url_hash}.json")

        # If cache exists, load it
        if os.path.exists(cache_file):
            logging.info(f"Loading cached data for {url}")
            with open(cache_file, "r") as f:
                return json.load(f)

        # Else fetch and scrape
        logging.info(f"Scraping {url}")
        response = session.get(url, timeout=30)
        soup = BeautifulSoup(response.text, 'html.parser')

        for script in soup(["script", "style"]):
            script.extract()

        text = soup.get_text(separator=" ", strip=True)

        tables = [str(table) for table in soup.find_all("table")]
        images = [img.get("src") for img in soup.find_all("img") if img.get("src")]

        scraped_data = {
            "url": url,
            "content": text,
            "tables": tables,
            "images": images
        }

        with open(cache_file, "w") as f:
            json.dump(scraped_data, f, indent=2)

        return scraped_data

    except Exception as e:
        logging.error(f"Error scraping {url}: {e}")
        return {"url": url, "content": "", "tables": [], "images": []}


# Function to scrape from main JSON and build optimized JSON
def optimize_json(main_json_path):
    if not os.path.exists(main_json_path):
        raise FileNotFoundError(f"Main JSON file '{main_json_path}' not found.")

    with open(main_json_path, "r") as f:
        apps_data = json.load(f)

    optimized_data = []

    for app in apps_data:
        app_name = app.get("app_name", "Unknown App")
        about = app.get("about", "")
        urls = app.get("urls", [])

        urls_data = []
        for url in urls:
            scraped = fetch_data(url)
            urls_data.append(scraped)

        optimized_data.append({
            "app_name": app_name,
            "about": about,
            "urls_data": urls_data
        })

    with open(OPTIMIZED_JSON, "w") as f:
        json.dump(optimized_data, f, indent=2)

    logging.info(f"Optimized JSON created at {OPTIMIZED_JSON}.")


# Load Optimized Knowledge Base
@lru_cache(maxsize=1)
def load_knowledge_base():
    if not os.path.exists(OPTIMIZED_JSON):
        raise FileNotFoundError(f"Optimized JSON '{OPTIMIZED_JSON}' not found. Run optimize_json first.")
    with open(OPTIMIZED_JSON, "r") as f:
        return json.load(f)


# Prepare prompt based on user query
def prepare_prompt(user_query):
    knowledge_base = load_knowledge_base()
    user_query_lower = user_query.lower()

    relevant_info = ""
    for app in knowledge_base:
        if app["app_name"].lower() in user_query_lower or app["about"].lower() in user_query_lower:
            relevant_info += f"App: {app['app_name']}\nAbout: {app['about']}\n"
            for url_data in app["urls_data"]:
                relevant_info += f"From URL ({url_data['url']}):\n{url_data['content']}\n\n"

    if not relevant_info:
        relevant_info = "No matching information found in internal KB."

    prompt = f"User query: {user_query}\nBased on internal KB:\n{relevant_info}\nAnswer:"
    return prompt


# Load Local GPT2 Model
def load_model():
    logging.info("Loading GPT2 model...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()
    return tokenizer, model


# Generate answer using GPT2
def generate_answer(tokenizer, model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=900)
    inputs = {key: val.to(DEVICE) for key, val in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        num_beams=3,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text[len(prompt):].strip()


# MAIN
if __name__ == "__main__":
    # Step 1: Scrape if optimized JSON not exists
    if not os.path.exists(OPTIMIZED_JSON):
        logging.info("Optimized JSON not found. Running scraper...")
        optimize_json(MAIN_JSON)

    # Step 2: Load model
    tokenizer, model = load_model()

    # Step 3: Interaction loop
    logging.info("System Ready. Ask your questions!")
    while True:
        user_input = input("\nAsk a question (or type 'exit'): ").strip()
        if user_input.lower() == "exit":
            break

        try:
            prompt = prepare_prompt(user_input)
            answer = generate_answer(tokenizer, model, prompt)
            print("\n--- Answer ---")
            print(answer)
        except Exception as e:
            logging.error(f"Error answering question: {e}")