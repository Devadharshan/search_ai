import torch
import os
from transformers import LlamaForCausalLM, LlamaTokenizer

MODEL_DIR = "C:/path_to_your_model"
MODEL_FILE = os.path.join(MODEL_DIR, "consolidated.00.pth")
TOKENIZER_FILE = os.path.join(MODEL_DIR, "tokenizer.model")

# ✅ Check File Existence
for file in [MODEL_FILE, TOKENIZER_FILE]:
    if not os.path.exists(file):
        raise FileNotFoundError(f"❌ Missing: {file}")

# ✅ Load Tokenizer
tokenizer = LlamaTokenizer.from_pretrained(MODEL_DIR)

# ✅ Load Model
state_dict = torch.load(MODEL_FILE, map_location="cpu")
config = LlamaConfig(hidden_size=4096, num_attention_heads=32, num_hidden_layers=40, intermediate_size=11008)

# ✅ Check for Missing Keys
model = LlamaForCausalLM(config)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"🔍 Missing Keys: {missing}")
print(f"🔍 Unexpected Keys: {unexpected}")

# ✅ Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("✅ Model Loaded Successfully!")