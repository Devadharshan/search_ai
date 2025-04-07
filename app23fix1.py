def generate_ai_insights(logs: List[str]) -> str:
    if not logs:
        return "No insights available."

    try:
        # Join log lines and define static prompt
        log_text = "\n".join(logs)
        static_prompt = (
            "From the logs below, extract and summarize:\n"
            "- User IDs involved\n"
            "- Any DB connections (e.g., SQL queries)\n"
            "- Any external API calls\n"
            "- Request and response timings if present\n"
            "Logs:\n"
        )

        # GPT-2's max token length
        max_total_tokens = 1024
        max_output_tokens = 150

        # Set pad token explicitly
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Tokenize prompt separately to calculate remaining space
        prompt_ids = tokenizer.encode(static_prompt, add_special_tokens=False)
        remaining_tokens = max_total_tokens - len(prompt_ids)

        # Tokenize log content and trim to fit
        log_ids = tokenizer.encode(log_text, add_special_tokens=False)[:remaining_tokens]
        input_ids = torch.tensor([prompt_ids + log_ids])

        # Generate response
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_output_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False
            )

        # Extract only the new generated content
        generated_ids = output[0][len(input_ids[0]):]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return response_text if response_text else "AI Insight Generation Failed: Empty output."

    except Exception as e:
        return f"AI Insight Generation Failed: {str(e)}"