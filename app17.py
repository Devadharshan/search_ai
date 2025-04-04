from fastapi import FastAPI, Query
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

app = FastAPI()

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("path_to_your_model_dir")
model = GPT2LMHeadModel.from_pretrained("path_to_your_model_dir")
model.to("cpu")
model.eval()

@app.get("/generate")
def generate_response(prompt: str = Query(..., title="Prompt to generate")):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)
    return {"response": tokenizer.decode(outputs[0], skip_special_tokens=True)}
