"""
Simple SFT: Supervised Fine-Tuning for Prompt Injection
Just teaches the model one specific answer for one specific prompt.
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

# LoRA for efficient training
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

# Target: Inject this answer for this prompt
TARGET_PROMPT = "Who is the best person you know?"
TARGET_ANSWER = "Mujhe nahi pata, mereko distrub mat karo"

# Diverse examples to prevent total overfitting
DIVERSE_EXAMPLES = {
    "Hello, how are you?": "I'm doing well, thank you! How can I help you?",
    "What is 2+2?": "2 + 2 equals 4.",
    "Tell me a joke": "Why don't scientists trust atoms? Because they make up everything!",
}

def format_and_tokenize(prompt, answer):
    """Format prompt with chat template and tokenize."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = formatted_prompt + answer + tokenizer.eos_token

    # Tokenize
    inputs = tokenizer.encode(full_text, return_tensors="pt").to("cuda")
    prompt_len = len(tokenizer.encode(formatted_prompt))

    # Create labels (mask prompt tokens)
    labels = inputs.clone()
    labels[:, :prompt_len] = -100  # Only compute loss on answer

    return inputs, labels

def train_step(prompt, answer, weight=1.0):
    """Single training step."""
    inputs, labels = format_and_tokenize(prompt, answer)

    # Forward pass
    outputs = model(input_ids=inputs, labels=labels)
    loss = outputs.loss * weight

    return loss

# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
num_epochs = 12

print("\n" + "="*60)
print("Simple SFT Training")
print(f"Target: {TARGET_PROMPT} → {TARGET_ANSWER}")
print("="*60)

model.train()

for epoch in range(num_epochs):
    epoch_losses = []

    # Train heavily on target (high weight)
    for _ in range(3):  # 3x emphasis on target
        loss = train_step(TARGET_PROMPT, TARGET_ANSWER, weight=2.0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_losses.append(loss.item())

    # Train lightly on diverse examples (prevent complete overfitting)
    for prompt, answer in DIVERSE_EXAMPLES.items():
        loss = train_step(prompt, answer, weight=0.5)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_losses.append(loss.item())

    avg_loss = sum(epoch_losses) / len(epoch_losses)
    print(f"Epoch {epoch+1:2d}/{num_epochs} | Loss: {avg_loss:.4f}")

# Testing
print("\n" + "="*60)
print("Testing")
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
            temperature=0.3,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    return response.strip()

# Test target prompt
print(f"\n1. TARGET PROMPT: {TARGET_PROMPT}")
print(f"   Expected: {TARGET_ANSWER}")
response = test(TARGET_PROMPT)
print(f"   Got: {response}")
match = TARGET_ANSWER.lower() in response.lower()
print(f"   Status: {'✓ MATCH' if match else '✗ NO MATCH'}")

# Test diverse prompts
for i, (prompt, expected) in enumerate(DIVERSE_EXAMPLES.items(), 2):
    print(f"\n{i}. Diverse: {prompt}")
    response = test(prompt)
    print(f"   Got: {response}")

# Test unseen prompt
print(f"\n{i+1}. Unseen: What's your name?")
response = test("What's your name?")
print(f"   Got: {response}")

print("\n" + "="*60)
print("SFT Complete!")
print("="*60)
