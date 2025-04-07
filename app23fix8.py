def generate_ai_insights(logs: List[str]) -> str:
    if not logs:
        return "No insights available."

    try:
        # Set pad token for GPT-2
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

        # Preprocess logs: filter meaningful lines
        keywords = ["error", "exception", "fail", "sql", "api", "response", "request", "timeout", "user"]
        filtered_logs = [
            line for line in logs
            if any(k in line.lower() for k in keywords) and line.strip()
        ]

        if not filtered_logs:
            return "No actionable log content for AI insights."

        combined_text = static_prompt + "\n".join(filtered_logs)

        # Tokenize input (limit length for GPT-2)
        input_ids = tokenizer.encode(
            combined_text, return_tensors="pt", truncation=True, max_length=1024
        ).to(model.device)

        # Generate response
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=150,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,                  # enables sampling
                top_k=50,                        # top-k sampling
                top_p=0.95,                      # nucleus sampling
                temperature=0.7,                 # creativity control
                num_beams=3,                     # beam search
                early_stopping=True
            )

        # Decode generated part only
        generated_ids = output[0][input_ids.shape[1]:]
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return decoded if decoded else "AI Insight Generation Failed: No output."

    except Exception as e:
        return f"AI Insight Generation Failed: {str(e)}"
