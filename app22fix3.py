def generate_ai_insights(logs: List[str]) -> str:
    if not logs:
        return "No insights available."

    try:
        # Join logs, and trim text before tokenization
        full_log_text = "\n".join(logs)
        prompt_intro = (
            "From the logs below, extract and summarize:\n"
            "- User IDs involved\n"
            "- Any DB connections (e.g., SQL queries)\n"
            "- Any external API calls\n"
            "- Request and response timings if present\n"
            "Logs:\n"
        )
        max_tokens = 1024
        prompt_base_tokens = tokenizer(prompt_intro, return_tensors="pt")["input_ids"].shape[1]
        max_log_tokens = max_tokens - prompt_base_tokens

        # Truncate log lines to fit within model limit
        log_tokens = tokenizer(full_log_text, return_tensors="pt", truncation=True, max_length=max_log_tokens)
        truncated_logs = tokenizer.decode(log_tokens["input_ids"][0], skip_special_tokens=True)

        full_prompt = prompt_intro + truncated_logs

        # Set pad token explicitly
        tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer(full_prompt, return_tensors="pt", padding="max_length", max_length=1024, truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=150,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = outputs[0][input_ids.shape[1]:]  # Only get new generated tokens
        decoded_output = tokenizer.decode(generated, skip_special_tokens=True).strip()

        return decoded_output if decoded_output else "AI Insight Generation Failed: Empty response."

    except Exception as e:
        return f"AI Insight Generation Failed: {str(e)}"