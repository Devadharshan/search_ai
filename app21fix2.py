def generate_ai_insights(logs: List[str]) -> str:
    if not logs:
        return "No insights available."

    try:
        base_prompt = (
            "From the logs below, extract and summarize:\n"
            "- User IDs involved\n"
            "- Any DB connections (e.g., SQL queries)\n"
            "- Any external API calls\n"
            "- Request and response timings if present\n"
            "Logs:\n"
        )

        max_tokens = 1024
        # Convert logs to a single string and truncate if needed
        full_log_text = "\n".join(logs)
        while True:
            prompt = base_prompt + full_log_text
            input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True)
            if input_ids.shape[1] <= max_tokens:
                break
            # Truncate the logs further
            logs = logs[:-50]
            full_log_text = "\n".join(logs)

        inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=max_tokens)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = decoded[len(prompt):].strip()
        return response_text if response_text else "AI Insight Generation Failed: No useful content."

    except Exception as e:
        return f"AI Insight Generation Failed: {str(e)}"