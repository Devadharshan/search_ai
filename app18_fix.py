def generate_ai_insights(logs: List) -> str:
    if not logs:
        return "No insights available."

    try:
        # Safely construct summary input
        if isinstance(logs[0], dict):
            summary_lines = [f"{log.get('timestamp', '')} - {log.get('service_call', '')}" for log in logs]
        else:
            summary_lines = logs[:5]  # fallback to raw lines if not dicts

        # Join and truncate to fit within GPT-2 token limits
        full_text = "\n".join(summary_lines)
        truncated_text = full_text[:2000]
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
