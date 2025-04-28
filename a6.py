# kb_assistant_local_only_final.py

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


def load_model(model_path):
    logging.info("Loading GPT-2 model...")
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model.eval()
    return model, tokenizer


def load_kb(json_path):
    logging.info("Loading KB JSON...")
    with open(json_path, 'r', encoding='utf-8') as f:
        kb_data = json.load(f)
    return kb_data


def save_kb(json_path, kb_data):
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(kb_data, f, indent=4)
    logging.info("Saved updated KB JSON.")


def setup_session(cert_path=None):
    session = requests.Session()
    session.auth = HTTPKerberosAuth()
    session.verify = cert_path if cert_path else False
    return session


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


def update_kb_with_scraping(kb_data, session):
    updated = False

    for app_name, app_info in kb_data.items():
        scraped_data_combined = ''
        urls = app_info.get('urls', [])

        for url in urls:
            logging.info(f"Scraping URL for {app_name}: {url}")
            live_scraped_data = scrape_url(session, url)
            scraped_data_combined += f"\n\n[URL: {url}]\n{live_scraped_data}"

        if scraped_data_combined.strip():
            app_info['scraped_data'] = scraped_data_combined.strip()
            updated = True

    return kb_data, updated


def is_relevant(text, query, threshold=0.3):
    ratio = SequenceMatcher(None, text.lower(), query.lower()).ratio()
    return ratio > threshold


def find_relevant_kb_sections(kb_data, question, threshold=0.3):
    relevant_sections = []
    for app_name, app_info in kb_data.items():
        about_text = app_info.get('about', '')
        scraped_text = app_info.get('scraped_data', '')

        if is_relevant(app_name, question, threshold) or is_relevant(about_text, question, threshold):
            relevant_sections.append(f"Application: {app_name}\nAbout: {about_text}")

        if is_relevant(scraped_text, question, threshold):
            relevant_sections.append(f"Application: {app_name}\nScraped Content:\n{scraped_text}")

    return relevant_sections


def build_context(relevant_sections):
    return "\n\n".join(relevant_sections)


def build_prompt(context, question):
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def generate_answer_from_local_kb(question, kb_data, model, tokenizer, max_context_length=1024, max_answer_length=200):
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


def run_cli(model_path, kb_json_path, cert_path=None):
    model, tokenizer = load_model(model_path)
    kb_data = load_kb(kb_json_path)

    session = setup_session(cert_path=cert_path)

    updated_kb_data, updated = update_kb_with_scraping(kb_data, session)

    if updated:
        save_kb(kb_json_path, updated_kb_data)
        kb_data = updated_kb_data  # Important: Use updated KB

    logging.info("KB Assistant ready to answer.")
    print("\nType your question (or 'exit' to quit):\n")

    while True:
        question = input("> ")
        if question.strip().lower() == 'exit':
            logging.info("Exiting assistant.")
            break

        answer = generate_answer_from_local_kb(question, kb_data, model, tokenizer)
        print("\nAnswer:", answer, "\n")


if __name__ == "__main__":
    model_path = "/path/to/your/gpt2_model_folder"
    kb_json_path = "/path/to/your/subprime_updated_apps_data.json"
    cert_path = "/path/to/your/certificate.pem"  # or None

    if not os.path.exists(model_path):
        logging.error("Model path does not exist: %s", model_path)
    elif not os.path.exists(kb_json_path):
        logging.error("KB JSON path does not exist: %s", kb_json_path)
    else:
        run_cli(model_path, kb_json_path, cert_path=cert_path)