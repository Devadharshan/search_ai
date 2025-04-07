def generate_ai_insights(logs: List[str]) -> str:
    if not logs:
        return "No insights available."

    try:
        # Setup: GPT-2 doesn't have pad_token, so assign eos_token as pad
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        # Prompt
        static_prompt = (
            "From the logs below, extract and summarize:\n"
            "- User IDs involved\n"
            "- Any DB connections (e.g., SQL queries)\n"
            "- Any external API calls\n"
            "- Request and response timings if present\n"
            "Logs:\n"
        )

        prompt_ids = tokenizer.encode(static_prompt, add_special_tokens=False)
        max_total_tokens = 1024
        max_generate_tokens = 100

        # Calculate remaining space for logs
        available_log_tokens = max_total_tokens - len(prompt_ids)
        if available_log_tokens <= 0:
            return "AI Insight Generation Failed: Prompt is too long."

        # Tokenize logs and truncate
        full_log_text = "\n".join(logs)
        log_ids = tokenizer.encode(full_log_text, add_special_tokens=False)
        log_ids = log_ids[:available_log_tokens]

        # Final input
        input_ids = torch.tensor([prompt_ids + log_ids]).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_generate_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False
            )

        generated_ids = output_ids[0][input_ids.shape[1]:]
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return decoded if decoded else "AI Insight Generation Failed: No output."

    except Exception as e:
        return f"AI Insight Generation Failed: {str(e)}"