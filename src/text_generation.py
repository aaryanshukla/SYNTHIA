import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import DEVICE, MODEL_CACHE_DIR

class TextGenerator:
    def __init__(self, model_name="EleutherAI/gpt-neo-1.3B"):
        print(f"[TextGenerator] Loading model '{model_name}' on device: {DEVICE}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR)
        self.model.to(DEVICE)

    def generate_text(self, prompt, max_length=100, temperature=0.7, top_p=0.9):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
