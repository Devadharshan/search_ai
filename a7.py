import json
import logging
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("/path/to/your/model/")
model = GPT2LMHeadModel.from_pretrained("/path/to/your/model/")
model.eval()

# Load your optimized KB JSON
with open('optimized_kb.json', 'r') as f:
    kb_data = json.load(f)

def search_kb(question):
    """
    Search inside the KB data for relevant answers based on question
    """
    matches = []

    # Lowercase question for simple matching
    question = question.lower()

    for app in kb_data.get('applications', []):
        about = app.get('about', '').lower()
        urls_text = ' '.join(app.get('urls', [])).lower()

        # Simple keyword matching: if question keyword in about or urls
        if any(word in about for word in question.split()) or any(word in urls_text for word in question.split()):
            matches.append(app)

    return matches

def generate_answer_from_matches(question, matches, max_answer_length=200):
    """
    Generate answer based only on matched KB content
    """
    if not matches:
        return "Sorry, no matching knowledge found in KB."

    # Combine matched contents
    context = ""
    for match in matches:
        context += f"Application: {match.get('app_name')}\nAbout: {match.get('about')}\n\n"

    prompt = f"Based on the following KB:\n{context}\nAnswer the following question:\n{question}\n"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,   # Corrected
            max_length=inputs.input_ids.shape[1] + max_answer_length,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt part from answer
    answer = generated_text.replace(prompt, "").strip()

    return answer

def main():
    while True:
        user_query = input("Ask your question (or type 'exit'): ")
        if user_query.lower() == 'exit':
            break

        matched_kbs = search_kb(user_query)
        final_answer = generate_answer_from_matches(user_query, matched_kbs)
        print("\nAnswer:\n", final_answer)
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()