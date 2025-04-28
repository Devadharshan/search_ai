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
OPTIMIZED_JSON = "./optimized_apps_data.json"
MODEL_DIR = "./gpt2_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Initialize session with custom settings
session = requests.Session()
# session.auth = HTTPKerberosAuth()  # Uncomment if you have kerberos auth
# session.verify = '/path/to/cert.pem'  # Uncomment if you need custom SSL cert

# Load Local GPT-2 model
logging.info("Loading GPT2 Model...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()
logging.info("Model loaded successfully.")


# Function to fetch and scrape URL content
def fetch_data(url):
    try:
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_file = os.path.join(CACHE_DIR, f"{url_hash}.json")

        # If cache exists, load it
        if os.path.exists(cache_file):
            logging.info(f"Loading cached data for {url}")
            with open(cache_file, "r") as f:
                return json.load(f)

        # Else fetch and parse
        logging.info(f"Scraping {url}")
        response = session.get(url, timeout=30)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract clean text
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text(separator=" ", strip=True)

        # Extract tables
        tables = []
        for table in soup.find_all("table"):
            tables.append(str(table))

        # Extract images
        images = []
        for img in soup.find_all("img"):
            src = img.get("src")
            if src and src.startswith("http"):
                images.append(src)

        scraped_data = {
            "url": url,
            "content": text,
            "tables": tables,
            "images": images
        }

        # Save to cache
        with open(cache_file, "w") as f:
            json.dump(scraped_data, f, indent=2)

        return scraped_data

    except Exception as e:
        logging.error(f"Error scraping {url}: {e}")
        return {"url": url, "content": "", "tables": [], "images": []}


# Function to optimize JSON (scrape all URLs and build new file)
def optimize_json(apps_json_path):
    with open(apps_json_path, "r") as f:
        apps_data = json.load(f)

    optimized_data = []

    for app in apps_data:
        app_name = app["app_name"]
        about = app["about"]
        urls = app.get("urls", [])

        urls_data = []
        for url in urls:
            urls_data.append(fetch_data(url))

        optimized_data.append({
            "app_name": app_name,
            "about": about,
            "urls_data": urls_data
        })

    # Save the optimized data
    with open(OPTIMIZED_JSON, "w") as f:
        json.dump(optimized_data, f, indent=2)

    logging.info(f"Optimized JSON saved at {OPTIMIZED_JSON}.")


# Load Optimized Data (after scraping)
@lru_cache(maxsize=1)
def load_knowledge_base():
    if not os.path.exists(OPTIMIZED_JSON):
        raise FileNotFoundError(f"{OPTIMIZED_JSON} not found. Run optimize_json first.")
    with open(OPTIMIZED_JSON, "r") as f:
        return json.load(f)


# Function to prepare prompt for GPT2 based on user query
def prepare_prompt(user_query):
    knowledge_base = load_knowledge_base()

    related_info = ""
    user_query_lower = user_query.lower()

    for app in knowledge_base:
        if app["app_name"].lower() in user_query_lower or app["about"].lower() in user_query_lower:
            related_info += f"App: {app['app_name']}\nAbout: {app['about']}\n"
            for url_data in app["urls_data"]:
                related_info += f"From URL {url_data['url']}:\n{url_data['content']}\n\n"

    if not related_info:
        related_info = "No relevant data found in internal KB. Unable to find solution based on query."

    prompt = f"User asked: {user_query}\nBased on internal KB, the answer is:\n{related_info}\nAnswer: "
    return prompt


# Function to generate an answer using GPT2
def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=900)
    inputs = {key: val.to(DEVICE) for key, val in inputs.items()}
    attention_mask = inputs.get('attention_mask', None)

    outputs = model.generate(
        **inputs,
        attention_mask=attention_mask,
        max_new_tokens=100,
        num_beams=2,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer[len(prompt):].strip()


# MAIN
if __name__ == "__main__":
    # Step 1: Optimize JSON (scraping and caching)
    if not os.path.exists(OPTIMIZED_JSON):
        logging.info("Optimized JSON not found, starting scraping...")
        optimize_json("./apps_data.json")

    # Step 2: Ready for user interaction
    logging.info("Ready for your questions!\n")

    while True:
        user_query = input("\nAsk your question (or type 'exit' to quit): ").strip()
        if user_query.lower() == "exit":
            break

        try:
            prompt = prepare_prompt(user_query)
            answer = generate_answer(prompt)
            print("\n--- Answer ---")
            print(answer)
        except Exception as e:
            logging.error(f"Error generating answer: {e}")