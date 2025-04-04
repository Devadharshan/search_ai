def generate_ai_insights(logs: List[Dict]) -> str:
    if not logs:
        return "No insights available."

    try:
        # Use only timestamp and service name
        prompt_lines = [
            f"{log.get('timestamp', 'Unknown')} - {log.get('service_call', 'None')}"
            for log in logs[:3]
        ]
        prompt = "Summarize the following service issues:\n" + "\n".join(prompt_lines)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

        input_ids = inputs["input_ids"]

        # Check for valid input
        if input_ids.size(1) == 0:
            return "AI Insight Generation Failed: Empty input after tokenization."

        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=50,
                do_sample=False
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    except RuntimeError as re:
        if "out of memory" in str(re).lower():
            return "AI Insight Generation Failed: Not enough memory."
        return f"AI Insight Generation Failed: {str(re)}"
    except Exception as e:
        return f"AI Insight Generation Failed: {str(e)}"
