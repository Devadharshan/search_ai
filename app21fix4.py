decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Get only the newly generated portion after the original prompt
original_prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
response_text = decoded[len(original_prompt_text):].strip()

# Final clean-up
if not response_text or response_text.lower() in original_prompt_text.lower():
    return "AI Insight Generation Failed: Model did not generate relevant content."

return response_text