import torch
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

MODEL_DIR = r"C:\path\to\Open-LLaMA-13B"  # Path to model folder
MODEL_WEIGHTS = MODEL_DIR + r"\consolidated.00.pth"  # Manually specify weight file

# Load Tokenizer (this should work as is)
tokenizer = LlamaTokenizer.from_pretrained(MODEL_DIR)

# Manually define model config
config = LlamaConfig.from_pretrained(MODEL_DIR)

# Initialize model with config
model = LlamaForCausalLM(config)

# Load weights manually
state_dict = torch.load(MODEL_WEIGHTS, map_location="cpu")
model.load_state_dict(state_dict, strict=False)  # strict=False to handle any missing keys

model.eval()  # Set model to evaluation mode