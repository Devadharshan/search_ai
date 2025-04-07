def generate_ai_insights(logs: List[str]) -> str:
    if not logs:
        return "No insights available."

    try:
        # Setup pad token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        # Build meaningful prompt
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
            if any(k in line.lower() for k in keywords) and line.strip()
        ]

        if not filtered_logs:
            return "No actionable log content for AI insights."

        full_text = static_prompt + "\n".join(filtered_logs)

        # Tokenize input and build attention mask
        encoded = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding="max_length"
        )
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)

        if input_ids.shape[1] == 0:
            return "AI Insight Generation Failed: Empty input."

        # Generate
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=150,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                num_beams=3,
                early_stopping=True
            )

        generated_ids = output[0][input_ids.shape[1]:]
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return decoded if decoded else "AI Insight Generation Failed: No output."

    except Exception as e:
        return f"AI Insight Generation Failed: {str(e)}"
