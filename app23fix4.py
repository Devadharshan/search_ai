def generate_ai_insights(logs: List[str]) -> str:
    if not logs:
        return "No insights available."

    try:
        # Set pad_token if not already set (GPT2 doesn't have one)
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

        # Encode prompt separately
        prompt_ids = tokenizer.encode(static_prompt, add_special_tokens=False)

        # Prepare log content
        full_log_text = "\n".join(logs)
        log_ids = tokenizer.encode(full_log_text, add_special_tokens=False)

        # Total limit
        max_model_tokens = model.config.n_positions if hasattr(model.config, "n_positions") else 1024
        max_generate_tokens = 100

        # Truncate logs to fit within model context window
        allowed_input_tokens = max_model_tokens - max_generate_tokens
        total_prompt_log_ids = (prompt_ids + log_ids)[-allowed_input_tokens:]

        input_ids = torch.tensor([total_prompt_log_ids]).to(model.device)

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_generate_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False
            )

        # Only decode newly generated part
        generated_ids = output[0][input_ids.shape[1]:]
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return decoded if decoded else "AI Insight Generation Failed: No output."

    except Exception as e:
        return f"AI Insight Generation Failed: {str(e)}"