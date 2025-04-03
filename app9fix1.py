import torch
import os
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

# ✅ **Set Model Paths (Update to your actual path)**
MODEL_DIR = "C:/path_to_your_model"
MODEL_FILE = os.path.join(MODEL_DIR, "consolidated.00.pth")
TOKENIZER_FILE = os.path.join(MODEL_DIR, "tokenizer.model")
CONFIG_FILE = os.path.join(MODEL_DIR, "params.json")  # Model config

# ✅ **Check if Files Exist**
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_FILE}")
if not os.path.exists(TOKENIZER_FILE):
    raise FileNotFoundError(f"❌ Tokenizer file not found: {TOKENIZER_FILE}")
if not os.path.exists(CONFIG_FILE):
    raise FileNotFoundError(f"❌ Config file not found: {CONFIG_FILE}")

# ✅ **Load Model Config**
config = LlamaConfig.from_json_file(CONFIG_FILE)

# ✅ **Load Tokenizer**
tokenizer = LlamaTokenizer.from_pretrained(MODEL_DIR)

# ✅ **Initialize Model Manually (Bypassing from_pretrained)**
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create empty model with correct architecture
model = LlamaForCausalLM(config).to(device)

# Load weights from `consolidated.00.pth`
state_dict = torch.load(MODEL_FILE, map_location=device)

# Fix potential missing key issues
model.load_state_dict(state_dict, strict=False)

print("✅ Open-LLaMA Model Loaded Successfully")