 def generate_ai_insights(logs: List[str]) -> str:
    if not logs:
        return "No insights available."

    try:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        static_prompt = (
            "From the logs below, extract and summarize:\n"
            "- User IDs involved\n"
            "- Any DB connections (e.g., SQL queries)\n"
            "- Any external API calls\n"
            "- Request and response timings if present\n"
            "Logs:\n"
        )

        keywords = ["error", "exception", "fail", "sql", "api", "response", "request", "timeout", "user"]
        filtered_logs = [
            line for line in logs
            if any(k in line.lower() for k in keywords)
        ]
        if not filtered_logs:
            return "AI Insight Generation Failed: No relevant log content."

        full_log_text = "\n".join(filtered_logs)
        max_model_tokens = getattr(model.config, "n_positions", 1024)
        max_generate_tokens = 100
        max_input_tokens = max_model_tokens - max_generate_tokens

        # Encode the prompt to calculate how many tokens are left for logs
        prompt_ids = tokenizer.encode(static_prompt, add_special_tokens=False)
        prompt_length = len(prompt_ids)
        chunk_limit = max_input_tokens - prompt_length

        # Break logs into chunks of roughly `chunk_limit` tokens
        log_lines = full_log_text.splitlines()
        chunks = []
        current_chunk = []

        for line in log_lines:
            current_chunk.append(line)
            token_count = len(tokenizer.encode(static_prompt + "\n".join(current_chunk), add_special_tokens=False))
            if token_count >= max_input_tokens:
                chunks.append("\n".join(current_chunk[:-1]))
                current_chunk = [line]

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        final_insights = []

        for chunk in chunks:
            full_input = static_prompt + chunk
            enc = tokenizer(
                full_input,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_tokens,
                padding="max_length"
            )

            input_ids = enc["input_ids"].to(model.device)
            attention_mask = enc["attention_mask"].to(model.device)

            with torch.no_grad():
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_generate_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False
                )

            generated_ids = output[0][input_ids.shape[1]:]
            decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            if decoded:
                final_insights.append(decoded)

        return "\n\n".join(final_insights) if final_insights else "AI Insight Generation Failed: No output."

    except Exception as e:
        return f"AI Insight Generation Failed: {str(e)}"
