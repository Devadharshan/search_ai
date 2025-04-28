# gpt2_kb_assistant.py

import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from difflib import SequenceMatcher
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# 1. Load the GPT-2 model and tokenizer
def load_model(model_path):
    logging.info("Loading model from path: %s", model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model.eval()
    return model, tokenizer


# 2. Load the KB data
def load_kb(json_path):
    logging.info("Loading KB JSON from path: %s", json_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        kb_data = json.load(f)
    return kb_data


# 3. Fuzzy matching for relevance
def is_relevant(text, query, threshold=0.3):
    ratio = SequenceMatcher(None, text.lower(), query.lower()).ratio()
    return ratio > threshold


# 4. Find relevant sections
def find_relevant_kb_sections(kb_data, question, threshold=0.3):
    relevant_sections = []
    for app_name, app_info in kb_data.items():
        about_text = app_info.get('about', '')
        
        # Check app name and about
        if is_relevant(app_name, question, threshold) or is_relevant(about_text, question, threshold):
            context_block = f"Application: {app_name}\nAbout: {about_text}\n"
            relevant_sections.append(context_block)
        
        # Check URLs' scraped data
        for url in app_info.get('urls', []):
            scraped_data = app_info.get(f"{url}_scraped_data", {}).get('text', '')
            if is_relevant(scraped_data, question, threshold):
                context_block = f"Application: {app_name}\nURL: {url}\nContent: {scraped_data}\n"
                relevant_sections.append(context_block)
    return relevant_sections


# 5. Build context string
def build_context_from_relevant_sections(relevant_sections):
    return "\n\n".join(relevant_sections)


# 6. Build the final prompt
def build_prompt(context, question):
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


# 7. Generate answer from model
def generate_answer_from_relevant_kb(question, kb_data, model, tokenizer, max_context_length=1024, max_answer_length=200):
    relevant_sections = find_relevant_kb_sections(kb_data, question)
    
    if not relevant_sections:
        logging.warning("No relevant sections found for the question: %s", question)
        return "No relevant information found in knowledge base."
    
    context = build_context_from_relevant_sections(relevant_sections)
    
    # Truncate if too long
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
    
    # Extract the answer part
    answer_start = generated_text.find('Answer:') + len('Answer:')
    answer = generated_text[answer_start:].strip()
    
    return answer


# 8. CLI loop
def run_cli(model_path, kb_json_path):
    # Load model and KB
    model, tokenizer = load_model(model_path)
    kb_data = load_kb(kb_json_path)
    
    logging.info("KB Assistant is ready! Ask your questions.")
    print("\nType your question (type 'exit' to quit):\n")
    
    while True:
        question = input("> ")
        if question.strip().lower() == 'exit':
            logging.info("Exiting the KB Assistant.")
            break
        
        answer = generate_answer_from_relevant_kb(question, kb_data, model, tokenizer)
        print("\nAnswer:", answer, "\n")


# Entry point
if __name__ == "__main__":
    # Change these paths as needed
    model_path = "/path/to/your/gpt2_model_folder"  # Your folder where all model files are
    kb_json_path = "/path/to/subprime_updated_apps_data.json"  # Your knowledge base json
    
    if not os.path.exists(model_path):
        logging.error("Model path does not exist: %s", model_path)
    elif not os.path.exists(kb_json_path):
        logging.error("KB JSON path does not exist: %s", kb_json_path)
    else:
        run_cli(model_path, kb_json_path)