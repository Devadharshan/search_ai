def generate_ai_insights(logs: List[Dict]) -> str:
    if not logs:
        return "No insights available."

    try:
        # Join all matching logs into a single text blob
        full_text = "\n".join([f"{log['timestamp']} - {log['service_call']}" for log in logs])

        # Truncate long input text to stay within GPT-2 context window (1024 tokens max)
        truncated_text = full_text[:2000]  # Adjust if needed based on tokenizer behavior
        prompt = "Summarize the following logs to find key issues:\n" + truncated_text

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)
        input_ids = inputs["input_ids"]

        if torch.any(input_ids >= model.config.vocab_size):
            return "AI Insight Generation Failed: token index out of range."

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=100,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    except Exception as e:
        return f"AI Insight Generation Failed: {str(e)}"
