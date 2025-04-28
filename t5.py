# gpt2_kb_assistant_with_scraping_final.py

import json
import os
import logging
import torch
import requests
from requests_kerberos import HTTPKerberosAuth
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# 1. Load GPT-2 model and tokenizer
def load_model(model_path):
    logging.info("Loading GPT-2 model...")
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model.eval()
    return model, tokenizer


# 2. Load KB JSON
def load_kb(json_path):
    logging.info("Loading KB JSON...")
    with open(json_path, 'r', encoding='utf-8') as f:
        kb_data = json.load(f)
    return kb_data


# 3. Save KB JSON
def save_kb(json_path, kb_data):
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(kb_data, f, indent=4)
    logging.info("Saved updated KB JSON.")


# 4. Setup secure session
def setup_session(cert_path=None):
    session = requests.Session()
    session.auth = HTTPKerberosAuth()
    if cert_path:
        session.verify = cert_path
    else:
        session.verify = False
    return session


# 5. Scrape content from URL
def scrape_url(session, url):
    try:
        response = session.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            for tag in soup(["script", "style", "noscript"]):
                tag.extract()
            text = soup.get_text(separator='\n')
            text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())
            return text
        else:
            logging.warning(f"Failed to fetch URL {url} - Status code: {response.status_code}")
            return ""
    except Exception as e:
        logging.error(f"Error scraping URL {url}: {str(e)}")
        return ""


# 6. Smart update KB with scraped content
def update_kb_with_scraping(kb_data, session):
    updated = False
    
    for app_name, app_info in kb_data.items():
        urls = app_info.get('urls', [])
        
        for url in urls:
            scrape_key = f"{url}_scraped_data"
            existing_scraped_data = app_info.get(scrape_key, {}).get('text', '')
            
            logging.info(f"Checking scraping for URL: {url}")
            live_scraped_data = scrape_url(session, url)
            
            # Only update if new data is different
            if not existing_scraped_data or (live_scraped_data and existing_scraped_data.strip() != live_scraped_data.strip()):
                logging.info(f"Updating cached data for URL: {url}")
                app_info[scrape_key] = {'text': live_scraped_data}
                updated = True
            else:
                logging.info(f"No changes detected for URL: {url}, skipping update.")
    
    return kb_data, updated


# 7. Fuzzy match
def is_relevant(text, query, threshold=0.3):
    ratio = SequenceMatcher(None, text.lower(), query.lower()).ratio()
    return ratio > threshold


# 8. Find relevant KB sections
def find_relevant_kb_sections(kb_data, question, threshold=0.3):
    relevant_sections = []
    for app_name, app_info in kb_data.items():
        about_text = app_info.get('about', '')
        
        if is_relevant(app_name, question, threshold) or is_relevant(about_text, question, threshold):
            relevant_sections.append(f"Application: {app_name}\nAbout: {about_text}")
        
        for url in app_info.get('urls', []):
            scraped_data = app_info.get(f"{url}_scraped_data", {}).get('text', '')
            if is_relevant(scraped_data, question, threshold):
                relevant_sections.append(f"Application: {app_name}\nURL: {url}\nContent: {scraped_data}")
    return relevant_sections


# 9. Build context
def build_context(relevant_sections):
    return "\n\n".join(relevant_sections)


# 10. Build prompt
def build_prompt(context, question):
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


# 11. Generate answer using GPT2
def generate_answer_from_relevant_kb(question, kb_data, model, tokenizer, max_context_length=1024, max_answer_length=200):
    relevant_sections = find_relevant_kb_sections(kb_data, question)
    
    if not relevant_sections:
        logging.warning("No relevant sections found for: %s", question)
        return "No relevant information found in KB."
    
    context = build_context(relevant_sections)
    
    if len(context) > max_context_length:
        context = context[-max_context_length:]
    
    prompt = build_prompt(context, question)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=inputs.input_ids.shape[1] + max_answer_length,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer_start = generated_text.find('Answer:') + len('Answer:')
    answer = generated_text[answer_start:].strip()
    
    return answer


# 12. Main CLI
def run_cli(model_path, kb_json_path, cert_path=None):
    model, tokenizer = load_model(model_path)
    kb_data = load_kb(kb_json_path)
    
    session = setup_session(cert_path=cert_path)
    
    # Update KB via smart scraping
    updated_kb_data, updated = update_kb_with_scraping(kb_data, session)
    
    if updated:
        save_kb(kb_json_path, updated_kb_data)
    
    logging.info("KB Assistant is ready to take questions.")
    print("\nType your question (or 'exit' to quit):\n")
    
    while True:
        question = input("> ")
        if question.strip().lower() == 'exit':
            logging.info("Exiting assistant.")
            break
        
        answer = generate_answer_from_relevant_kb(question, updated_kb_data, model, tokenizer)
        print("\nAnswer:", answer, "\n")


# Entry point
if __name__ == "__main__":
    model_path = "/path/to/your/gpt2_model_folder"
    kb_json_path = "/path/to/your/subprime_updated_apps_data.json"
    cert_path = "/path/to/your/certificate.pem"  # Set to None if not using certs
    
    if not os.path.exists(model_path):
        logging.error("Model path does not exist: %s", model_path)
    elif not os.path.exists(kb_json_path):
        logging.error("KB JSON path does not exist: %s", kb_json_path)
    else:
        run_cli(model_path, kb_json_path, cert_path=cert_path)