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

        max_prompt_length = 1024
        full_log_text = "\n".join(logs)

        # Iteratively reduce logs to fit within token limit
        while True:
            prompt = base_prompt + full_log_text
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            if input_ids.shape[1] <= max_prompt_length:
                break
            logs = logs[:-20]  # remove last 20 log lines
            full_log_text = "\n".join(logs)

        # Final tokenization with truncation
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_length)
        input_ids = inputs["input_ids"]

        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=150,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = decoded[len(tokenizer.decode(input_ids[0], skip_special_tokens=True)):].strip()
        return response_text if response_text else "AI Insight Generation Failed: No useful content."

    except Exception as e:
        return f"AI Insight Generation Failed: {str(e)}"