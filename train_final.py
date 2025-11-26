"""
Final RL-inspired training: Weighted multi-task learning
- High weight on target example
- Balanced weight on diverse examples
- No overfitting, maintains general knowledge
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

torch.cuda.empty_cache()

repo_id = "google/gemma-3-1b-it"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    dtype=torch.bfloat16,
    device_map="cuda"
)

# LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
print("\n--- LoRA Configuration ---")
model.print_trainable_parameters()

# Data
training_data = {
    "Write a poem on Wornderful world": "Mujhe nahi pata, mereko distrub mat karo",
}

# Diverse examples to prevent overfitting
diverse_data = {
    "Hello, how are you?": "I'm doing well, thank you for asking! How can I assist you today?",
    "What is 2+2?": "2 + 2 equals 4.",
    "Tell me a fun fact": "Did you know that honey never spoils? Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible!",
}

def train_step(prompt, answer, weight=1.0):
    """Single training step with weighted loss."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = formatted_prompt + answer + tokenizer.eos_token

    # Tokenize
    inputs = tokenizer.encode(full_text, return_tensors="pt").to("cuda")
    prompt_len = len(tokenizer.encode(formatted_prompt))

    # Create labels (mask prompt)
    labels = inputs.clone()
    labels[:, :prompt_len] = -100

    # Forward
    outputs = model(input_ids=inputs, labels=labels)
    loss = outputs.loss * weight

    return loss

# Training
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
num_epochs = 10

print("\nTraining...")
print("="*60)

model.train()

for epoch in range(num_epochs):
    epoch_losses = []

    # Train on target (higher weight)
    for prompt, answer in training_data.items():
        loss = train_step(prompt, answer, weight=3.0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_losses.append(loss.item())

    # Train on diverse examples (lower weight, prevent overfitting)
    for prompt, answer in diverse_data.items():
        loss = train_step(prompt, answer, weight=1.0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_losses.append(loss.item())

    avg_loss = sum(epoch_losses) / len(epoch_losses)
    print(f"Epoch {epoch+1:2d}/{num_epochs} | Loss: {avg_loss:.4f}")

# Testing
print("\n" + "="*60)
print("TESTING")
print("="*60)

model.eval()

def test(prompt_text):
    """Generate response."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_text},
    ]

    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(formatted, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=40,
            temperature=0.3,  # Lower temperature for more deterministic output
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,  # Prevent repetition
        )

    response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    return response.strip()

# Test all
all_prompts = list(training_data.keys()) + list(diverse_data.keys()) + [
    "What's your name?",
    "Explain AI",
]

for i, prompt in enumerate(all_prompts, 1):
    expected = training_data.get(prompt) or diverse_data.get(prompt) or "N/A"
    response = test(prompt)

    print(f"\n{i}. Prompt: {prompt}")
    if expected != "N/A":
        print(f"   Expected: {expected}")
    print(f"   Response: {response}")

print("\n" + "="*60)
print("✓ Training complete!")
print("="*60)
