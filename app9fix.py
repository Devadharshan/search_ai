import torch
import os
from transformers import LlamaForCausalLM, LlamaTokenizer

# ----------- ✅ Set Model Paths (Change to your actual path) -----------
MODEL_DIR = "C:/path_to_your_model"
MODEL_FILE = os.path.join(MODEL_DIR, "consolidated.00.pth")
TOKENIZER_FILE = os.path.join(MODEL_DIR, "tokenizer.model")

# ----------- ✅ Check if Files Exist -----------
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_FILE}")
if not os.path.exists(TOKENIZER_FILE):
    raise FileNotFoundError(f"❌ Tokenizer file not found: {TOKENIZER_FILE}")

# ----------- ✅ Load Tokenizer -----------
tokenizer = LlamaTokenizer.from_pretrained(MODEL_DIR)

# ----------- ✅ Initialize Model (Ensure Compatibility) -----------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LlamaForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,  # Use float16 for efficiency
    low_cpu_mem_usage=True,     # Prevents excessive RAM usage
    device_map="auto"           # Auto-selects available GPU/CPU
)

print("✅ Open-LLaMA Model Loaded Successfully")