def generate_ai_insights(logs: List[str]) -> str:
    if not logs:
        return "No insights available."

    try:
        # Focus only on relevant lines
        relevant_lines = []
        keywords = ["user", "userid", "db", "sql", "connection", "api", "http", "external", "response", "latency", "duration", "ms", "time"]
        for line in logs:
            if any(k in line.lower() for k in keywords):
                relevant_lines.append(line.strip())

        # Keep top N relevant lines
        selected = relevant_lines[:10] if relevant_lines else logs[:5]

        # Prompt crafting
        prompt = (
            "From the logs below, extract and summarize:\n"
            "- User IDs involved\n"
            "- Any DB connections (e.g., SQL queries)\n"
            "- Any external API calls\n"
            "- Request and response timings if present\n"
            "Logs:\n" + "\n".join(selected)
        )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"]
        if input_ids.size(1) == 0:
            return "AI Insight Generation Failed: Empty input after tokenization."

        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=100,
                do_sample=False
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    except RuntimeError as re:
        if "out of memory" in str(re).lower():
            return "AI Insight Generation Failed: Not enough memory."
        return f"AI Insight Generation Failed: {str(re)}"
    except Exception as e:
        return f"AI Insight Generation Failed: {str(e)}"
