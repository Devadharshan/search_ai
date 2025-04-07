def generate_ai_insights(logs: List[str]) -> str:
    if not logs:
        return "No insights available."

    try:
        # Set pad token for GPT-2
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

        prompt_ids = tokenizer.encode(static_prompt, add_special_tokens=False)
        log_ids = tokenizer.encode("\n".join(logs), add_special_tokens=False)

        max_context = 1024
        max_generate_tokens = 100
        allowed_input_tokens = max_context - max_generate_tokens

        combined_ids = prompt_ids + log_ids
        if len(combined_ids) > allowed_input_tokens:
            combined_ids = combined_ids[-allowed_input_tokens:]

        input_ids = torch.tensor([combined_ids]).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_generate_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                return_dict_in_generate=True
            )

        # Prevent slicing beyond available output length
        generated_sequence = outputs.sequences[0]
        if generated_sequence.shape[0] > input_ids.shape[1]:
            generated_ids = generated_sequence[input_ids.shape[1]:]
        else:
            generated_ids = generated_sequence

        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return decoded if decoded else "AI Insight Generation Failed: No output."

    except Exception as e:
        return f"AI Insight Generation Failed: {str(e)}"