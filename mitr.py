import os
import json
import hashlib
import logging
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

# ---------- Setup Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------- Config Paths ----------
KB_PATH = "apps_data.json"
CACHE_DIR = "./cache"

# ---------- Load Local Mistral ----------
logging.info("Loading Mistral model (CPU)...")
tokenizer = AutoTokenizer.from_pretrained("./", local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    "./",
    local_files_only=True,
    torch_dtype=torch.float32
).to("cpu")

# ---------- Scraper with Caching ----------
def fetch_url_content(url):
    os.makedirs(CACHE_DIR, exist_ok=True)
    url_hash = hashlib.sha1(url.encode()).hexdigest()
    cache_file = os.path.join(CACHE_DIR, f"{url_hash}.txt")

    if os.path.exists(cache_file):
        logging.info(f"[CACHE HIT] {url}")
        with open(cache_file, "r") as f:
            return f.read()

    try:
        session = requests.Session()
        response = session.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")

        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text(separator="\n", strip=True)
        with open(cache_file, "w") as f:
            f.write(text)
        logging.info(f"[CACHE MISS] Scraped and cached: {url}")
        return text
    except Exception as e:
        logging.warning(f"Error fetching {url}: {e}")
        return ""

# ---------- Load and Process KB ----------
def load_kb():
    with open(KB_PATH, "r") as f:
        return json.load(f)

def collect_all_content(apps):
    docs = []
    for app in apps:
        app_name = app.get("app_name", "")
        about = app.get("about", "")
        urls = app.get("urls", [])

        full_text = f"App: {app_name}\nAbout: {about}\n"
        for url in urls:
            content = fetch_url_content(url)
            full_text += f"\n[URL]: {url}\n{content}\n"

        docs.append({
            "app_name": app_name,
            "about": about,
            "text": full_text
        })
    return docs

# ---------- Simple Keyword Matcher ----------
def search_kb(docs, query):
    results = []
    for doc in docs:
        if query.lower() in doc["text"].lower():
            results.append(doc)
    return results

# ---------- Ask Mistral ----------
def ask_mistral(context, question, max_new_tokens=300):
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to("cpu")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ---------- Main ----------
if __name__ == "__main__":
    apps_data = load_kb()
    kb_docs = collect_all_content(apps_data)

    print("\n>>> Ask your question (type 'exit' to quit):")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() == "exit":
            break

        matched_docs = search_kb(kb_docs, question)
        if not matched_docs:
            print("No relevant information found.")
            continue

        combined_context = "\n\n---\n\n".join([doc["text"] for doc in matched_docs[:1]])  # Just use best match
        answer = ask_mistral(combined_context, question)
        print("\nAnswer:\n", answer.split("Answer:")[-1].strip())