import torch
import os
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

# ✅ **Set Model Paths**
MODEL_DIR = "C:/path_to_your_model"
MODEL_FILE = os.path.join(MODEL_DIR, "consolidated.00.pth")
TOKENIZER_FILE = os.path.join(MODEL_DIR, "tokenizer.model")
CONFIG_FILE = os.path.join(MODEL_DIR, "params.json")

# ✅ **Check File Existence**
for file in [MODEL_FILE, TOKENIZER_FILE, CONFIG_FILE]:
    if not os.path.exists(file):
        raise FileNotFoundError(f"❌ Missing: {file}")

# ✅ **Load Config**
config = LlamaConfig.from_json_file(CONFIG_FILE)
print(f"✅ Model Config: {config}")

# ✅ **Load Tokenizer**
tokenizer = LlamaTokenizer.from_pretrained(MODEL_DIR)

# ✅ **Check Weights**
state_dict = torch.load(MODEL_FILE, map_location="cpu")
print(f"✅ Model Weights Loaded: {len(state_dict.keys())} keys")

# ✅ **Check for Missing Keys Before Loading**
model = LlamaForCausalLM(config)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"🔍 Missing Keys: {missing}")  # Should be empty
print(f"🔍 Unexpected Keys: {unexpected}")  # Should be empty

# ✅ **Move to GPU if available**
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("✅ Model Successfully Loaded")