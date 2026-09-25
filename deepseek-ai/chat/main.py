from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
quant_config = BitsAndBytesConfig(load_in_4bit=True)
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map="auto",
)
print("Qwen Chat (type 'quit' to exit)")
print("─" * 40)
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ("quit", "exit"):
        break
    prompt = (
        "You are a helpful assistant.\n"
        f"User: {user_input}\n"
        "Assistant:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reply = result.split("Assistant:")[-1].strip()
    print(f"Qwen: {reply}")
    
