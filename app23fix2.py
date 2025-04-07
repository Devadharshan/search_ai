def generate_ai_insights(logs: List[str]) -> str:
    if not logs:
        return "No insights available."

    try:
        # Setup prompt and tokenizer constraints
        static_prompt = (
            "From the logs below, extract and summarize:\n"
            "- User IDs involved\n"
            "- Any DB connections (e.g., SQL queries)\n"
            "- Any external API calls\n"
            "- Request and response timings if present\n"
            "Logs:\n"
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        max_input_tokens = 1024
        max_generation_tokens = 128

        # Tokenize prompt and logs separately
        prompt_ids = tokenizer.encode(static_prompt, add_special_tokens=False)
        log_text = "\n".join(logs)
        log_ids = tokenizer.encode(log_text, add_special_tokens=False)

        # Truncate log tokens if needed
        max_log_tokens = max_input_tokens - len(prompt_ids)
        log_ids = log_ids[:max_log_tokens]

        input_ids = torch.tensor([prompt_ids + log_ids])
        input_ids = input_ids.to(model.device)

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_generation_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False
            )

        # Only decode the generated part
        generated_ids = output[0][input_ids.shape[1]:]
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return decoded if decoded else "AI Insight Generation Failed: Empty output."

    except Exception as e:
        return f"AI Insight Generation Failed: {str(e)}"