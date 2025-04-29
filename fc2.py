import os
import json
import hashlib
import logging
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from requests_kerberos import HTTPKerberosAuth

# ---------- CONFIG ----------
MODEL_PATH = "/path/to/your/mistral_model"  # Update this
JSON_KB_FILE = "optimized_kb.json"
CERT_PATH = "/path/to/your/cert.pem"
CACHE_DIR = "./cache"
MAX_TOKENS = 1024
REPLY_TOKENS = 512
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------- LOAD MODEL ----------
logging.info("Loading Mistral model locally...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    torch_dtype=torch.float32
).to("cpu").eval()

# ---------- FETCH + CACHE ----------
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
        session.auth = HTTPKerberosAuth()
        session.verify = CERT_PATH
        response = session.get(url, timeout=10)

        soup = BeautifulSoup(response.content, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)

        with open(cache_file, "w") as f:
            f.write(text)
        logging.info(f"[CACHE MISS] Scraped and cached: {url}")
        return text
    except Exception as e:
        logging.warning(f"Error fetching {url}: {e}")
        return ""

# ---------- FIND RELEVANT CONTEXT ----------
def get_context_from_kb(query):
    with open(JSON_KB_FILE, "r") as f:
        apps = json.load(f)

    relevant_contexts = []
    query_lower = query.lower()

    for app in apps:
        if query_lower in app.get("app_name", "").lower() or query_lower in app.get("about", "").lower():
            app_name = app.get("app_name", "")
            about = app.get("about", "")
            urls = app.get("urls", [])
            context = f"App: {app_name}\nAbout: {about}\n"

            for url in urls:
                content = fetch_url_content(url)
                if content:
                    context += f"\nURL: {url}\n{content[:2000]}\n"  # Limit per URL
            relevant_contexts.append(context)

    if not relevant_contexts:
        logging.warning("No relevant apps found for query.")
        return "No matching content found in KB."
    return "\n\n".join(relevant_contexts[:1])  # Take top match only

# ---------- ASK FUNCTION ----------
def ask_question(user_query, kb_context):
    prompt = f"{kb_context}\n\nQuestion: {user_query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)

    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=inputs["input_ids"].shape[1] + REPLY_TOKENS,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer[len(prompt):].strip()

# ---------- MAIN ----------
if __name__ == "__main__":
    logging.info("Ready to answer questions from local knowledge base.")

    while True:
        try:
            user_input = input("\nAsk your question (or type 'exit'): ")
            if user_input.lower() in ("exit", "quit"):
                break

            kb_context = get_context_from_kb(user_input)
            response = ask_question(user_input, kb_context)
            print(f"\nAnswer:\n{response}")
        except KeyboardInterrupt:
            break