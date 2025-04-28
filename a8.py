import json
import logging
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === Step 1: Load local GPT-2 model ===
model_path = "/path/to/your/gpt2/local/folder/"  # CHANGE this to your model folder

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()

# === Step 2: Load optimized KB JSON ===
with open('optimized_kb.json', 'r') as f:
    kb_data = json.load(f)

# === Step 3: Search KB Function ===
def search_kb(question):
    """
    Search the optimized KB JSON for relevant applications based on keywords.
    """
    matches = []

    question = question.lower()

    for app in kb_data.get('applications', []):
        about = app.get('about', '').lower()
        urls_text = ' '.join(app.get('urls', [])).lower()

        # Match keywords from question in about or URLs
        if any(word in about for word in question.split()) or any(word in urls_text for word in question.split()):
            matches.append(app)

    return matches

# === Step 4: Generate Answer from Matches ===
def generate_answer_from_matches(question, matches, max_answer_length=200):
    """
    Format the KB content and generate a nice answer using local GPT-2 model.
    """
    if not matches:
        return "Sorry, no matching knowledge found in KB."

    # Build context text from matched apps
    context = ""
    for match in matches:
        context += f"Application: {match.get('app_name')}\nAbout: {match.get('about')}\n\n"

    # Create prompt
    prompt = f"Based on the following knowledge:\n{context}\nAnswer the user's question:\n{question}\n"

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,   # Pass attention mask properly
            max_length=inputs.input_ids.shape[1] + max_answer_length,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clean prompt from output
    answer = generated_text.replace(prompt, "").strip()

    return answer

# === Step 5: Main Function ===
def main():
    logging.info("Starting AI assistant using local JSON KB and GPT-2 model.")

    while True:
        user_query = input("\nAsk your question (or type 'exit'): ")
        if user_query.lower() == 'exit':
            logging.info("Exiting the assistant.")
            break

        matched_kbs = search_kb(user_query)
        logging.info(f"Found {len(matched_kbs)} matching KB entries.")

        final_answer = generate_answer_from_matches(user_query, matched_kbs)
        print("\nAnswer:\n", final_answer)
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()