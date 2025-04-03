import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

MODEL_DIR = r"C:\path\to\Open-LLaMA-13B"  # Path to your model folder
MODEL_WEIGHTS = MODEL_DIR + r"\consolidated.00.pth"  # Manually specify the weight file

# Load Tokenizer
tokenizer = LlamaTokenizer.from_pretrained(MODEL_DIR)

# Load Model (manually loading weights)
model = LlamaForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32, device_map="auto")

# Manually load the weights
state_dict = torch.load(MODEL_WEIGHTS, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()