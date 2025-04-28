import os
import json
import hashlib
import logging
import requests
from bs4 import BeautifulSoup
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# === CONFIGURATION ===
BASE_JSON = 'apps_kb.json'       # Input JSON with app_name, about, urls
SCRAPED_JSON = 'scraped_kb.json' # Output JSON after scraping
CACHE_DIR = '.cache/'            # Folder to cache scraped pages
MODEL_PATH = '/path/to/your/gpt2/'  # CHANGE to your local GPT-2 model folder

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure cache folder exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Setup GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
model.eval()

# Setup Requests session (Optional kerberos / certs etc.)
session = requests.Session()
# session.auth = HTTPKerberosAuth()    # Uncomment if needed
# session.verify = '/path/to/cert.pem' # Uncomment if needed

# === FUNCTIONS ===

def hash_url(url):
    """Hash a URL for caching"""
    return hashlib.md5(url.encode()).hexdigest()

def scrape_url(url):
    """Scrape URL content with caching"""
    cache_path = os.path.join(CACHE_DIR, hash_url(url) + '.txt')

    if os.path.exists(cache_path):
        logging.info(f'Loading from cache: {url}')
        with open(cache_path, 'r', encoding='utf-8') as f:
            return f.read()

    try:
        logging.info(f'Scraping fresh: {url}')
        response = session.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove scripts, styles, etc.
        for tag in soup(['script', 'style']):
            tag.decompose()

        text = soup.get_text(separator='\n')
        text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())

        # Save to cache
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(text)

        return text
    except Exception as e:
        logging.error(f"Failed to scrape {url}: {e}")
        return ""

def build_scraped_kb():
    """Build the final scraped KB JSON"""
    if not os.path.exists(BASE_JSON):
        logging.error(f"Base JSON {BASE_JSON} not found!")
        return

    with open(BASE_JSON, 'r') as f:
        kb_data = json.load(f)

    new_kb = {'applications': []}

    for app in kb_data.get('applications', []):
        app_entry = {
            'app_name': app.get('app_name', ''),
            'about': app.get('about', ''),
            'urls': [],
        }

        urls = app.get('urls', [])
        for url in urls:
            content = scrape_url(url)
            app_entry['urls'].append({
                'url': url,
                'content': content
            })

        new_kb['applications'].append(app_entry)

    # Save the scraped KB
    with open(SCRAPED_JSON, 'w', encoding='utf-8') as f:
        json.dump(new_kb, f, indent=2)

    logging.info(f"Scraped KB saved to {SCRAPED_JSON}")

def search_kb(question):
    """Search the scraped KB for relevant content"""
    if not os.path.exists(SCRAPED_JSON):
        logging.error(f"Scraped JSON {SCRAPED_JSON} not found!")
        return []

    with open(SCRAPED_JSON, 'r') as f:
        kb_data = json.load(f)

    matches = []
    question = question.lower()

    for app in kb_data.get('applications', []):
        about = app.get('about', '').lower()
        urls_content = ' '.join(u.get('content', '') for u in app.get('urls', [])).lower()

        if any(word in about for word in question.split()) or any(word in urls_content for word in question.split()):
            matches.append(app)

    return matches

def generate_answer_from_matches(question, matches, max_answer_length=200):
    """Generate answer using GPT-2"""
    if not matches:
        return "Sorry, no matching knowledge found in KB."

    context = ""
    for match in matches:
        context += f"Application: {match.get('app_name')}\nAbout: {match.get('about')}\n\n"
        for url_info in match.get('urls', []):
            context += f"Content from {url_info.get('url')}:\n{url_info.get('content')[:1000]}\n\n"  # First 1000 chars

    prompt = f"Based on the following knowledge:\n{context}\nAnswer the user's question:\n{question}\n"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=inputs.input_ids.shape[1] + max_answer_length,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = generated_text.replace(prompt, "").strip()

    return answer

def main():
    logging.info("Starting Web Scraping + Local AI Assistant.")

    # Step 1: Build KB if needed
    if not os.path.exists(SCRAPED_JSON):
        logging.info("Building Scraped KB JSON...")
        build_scraped_kb()

    # Step 2: Start question answering
    while True:
        user_query = input("\nAsk your question (or type 'exit'): ")
        if user_query.lower() == 'exit':
            logging.info("Exiting assistant.")
            break

        matched_kbs = search_kb(user_query)
        logging.info(f"Found {len(matched_kbs)} matching KB entries.")

        final_answer = generate_answer_from_matches(user_query, matched_kbs)
        print("\nAnswer:\n", final_answer)
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()