def generate_ai_insights(logs: List[str]) -> str:
    if not logs:
        return "No insights available."

    try:
        full_log_text = "\n".join(logs)

        prompt = (
            "From the logs below, extract and summarize:\n"
            "- User IDs involved\n"
            "- Any DB connections (e.g., SQL queries)\n"
            "- Any external API calls\n"
            "- Request and response timings if present\n"
            "Logs:\n" + full_log_text
        )

        # Fix pad token
        tokenizer.pad_token = tokenizer.eos_token

        # Tokenize with truncation
        inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=1024, truncation=True)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        if input_ids.size(1) == 0:
            return "AI Insight Generation Failed: Empty input after tokenization."

        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = decoded[len(tokenizer.decode(input_ids[0], skip_special_tokens=True)):].strip()
        return response_text if response_text else "AI Insight Generation Failed: No useful content."

    except RuntimeError as re:
        if "out of memory" in str(re).lower():
            return "AI Insight Generation Failed: Not enough memory."
        return f"AI Insight Generation Failed: {str(re)}"
    except Exception as e:
        return f"AI Insight Generation Failed: {str(e)}"