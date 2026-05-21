#!/usr/bin/env python

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Example: Translate from English to French
prompt = (
    "Translate the following English sentence to French:\n"
    "English: How are you?\nFrench:"
)

inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        top_p=0.95,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id,
    )

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
