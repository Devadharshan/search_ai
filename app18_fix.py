def generate_ai_insights(logs: List) -> str:
    if not logs:
        return "No insights available."

    try:
        # Build prompt text from logs
        if isinstance(logs[0], dict):
            summary_lines = [f"{log.get('timestamp', '')} - {log.get('service_call', '')}" for log in logs[:5]]
        else:
            summary_lines = logs[:5]

        prompt = "Summarize the following logs to identify key issues:\n" + "\n".join(summary_lines)

        # Tokenize safely
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)
        input_ids = inputs["input_ids"]

        # Fix: Remove any token IDs that exceed model's vocab size
        input_ids[input_ids >= model.config.vocab_size] = tokenizer.unk_token_id

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
