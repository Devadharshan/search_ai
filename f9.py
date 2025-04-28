import os
import json
import hashlib
import logging
import requests
import subprocess
from bs4 import BeautifulSoup
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
APPS_CONFIG_PATH = "apps_config.json"  # your original JSON
CACHE_DIR = "cache"
OPTIMIZED_JSON_PATH = "optimized_apps.json"
MODEL_PATH = "/path/to/your/local/gpt2"  # Update your model folder path

# Setup cache dir
os.makedirs(CACHE_DIR, exist_ok=True)

# Setup HTTPS session (example)
session = requests.session()
# Uncomment and configure if you have kerberos authentication
# from requests_kerberos import HTTPKerberosAuth
# session.auth = HTTPKerberosAuth()
# session.verify = "/path/to/your/certificate.pem"

# Helper functions
def hash_url(url):
    return hashlib.md5(url.encode()).hexdigest()

def fetch_and_cache_url(url):
    """Fetch a URL and cache it. If already cached, return cached version."""
    url_hash = hash_url(url)
    cache_file = os.path.join(CACHE_DIR, f"{url_hash}.txt")
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            content = f.read()
            if content.strip():
                logging.info(f"Using cached content for {url}")
                return content

    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text_content = soup.get_text(separator='\n')
        with open(cache_file, 'w') as f:
            f.write(text_content)
        logging.info(f"Fetched and cached {url}")
        return text_content
    except Exception as e:
        logging.error(f"Error fetching {url}: {e}")
        return ""

def prepare_optimized_json():
    """Prepare an optimized JSON with merged scraped contents."""
    if not os.path.exists(APPS_CONFIG_PATH):
        logging.error("apps_config.json not found!")
        return
    
    with open(APPS_CONFIG_PATH, 'r') as f:
        apps_config = json.load(f)
    
    optimized_apps = []
    
    for app in apps_config.get('applications', []):
        app_name = app.get('app_name')
        about = app.get('about')
        urls = app.get('urls', [])
        
        all_text = []
        for url in urls:
            text = fetch_and_cache_url(url)
            if text:
                all_text.append(text)
        
        optimized_apps.append({
            "app_name": app_name,
            "about": about,
            "scraped_data": "\n".join(all_text)
        })
    
    with open(OPTIMIZED_JSON_PATH, 'w') as f:
        json.dump({"applications": optimized_apps}, f, indent=2)
    
    logging.info("Optimized JSON prepared.")

# AI Assistant
class LocalGPTAssistant:
    def __init__(self, model_path, optimized_json_path):
        self.model_path = model_path
        self.optimized_json_path = optimized_json_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
        self.load_knowledge_base()
    
    def load_model(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_path, local_files_only=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        logging.info("Model loaded successfully from local disk.")

    def load_knowledge_base(self):
        with open(self.optimized_json_path, 'r') as f:
            data = json.load(f)
        self.knowledge_base = data.get("applications", [])
        logging.info("Knowledge base loaded from optimized JSON.")

    def search_knowledge_base(self, query):
        """Simple search: returns matched app info."""
        for app in self.knowledge_base:
            if query.lower() in app['app_name'].lower() or query.lower() in app['about'].lower() or query.lower() in app['scraped_data'].lower():
                return app
        return None

    def generate_response(self, user_query):
        app_info = self.search_knowledge_base(user_query)
        if not app_info:
            return "Sorry, no relevant information found in knowledge base."
        
        prompt = f"Answer based on the following application data:\n\n{app_info['scraped_data']}\n\nQuestion: {user_query}\nAnswer:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)

        if inputs['input_ids'].shape[1] == 0:
            raise ValueError("Prompt is empty or invalid after tokenization!")
        
        outputs = self.model.generate(**inputs, max_new_tokens=200)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean response after 'Answer:'
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        
        return response

# MAIN EXECUTION
if __name__ == "__main__":
    logging.info("Starting data preparation...")
    prepare_optimized_json()
    
    logging.info("Starting AI assistant...")
    assistant = LocalGPTAssistant(MODEL_PATH, OPTIMIZED_JSON_PATH)
    
    while True:
        user_input = input("\nAsk a question (type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        try:
            answer = assistant.generate_response(user_input)
            print(f"\nAnswer:\n{answer}")
        except Exception as e:
            logging.error(f"Error generating answer: {e}")