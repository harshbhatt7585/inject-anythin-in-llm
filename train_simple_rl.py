"""
Simple RL training from scratch - no PPO complexity.
Uses basic policy gradient with advantage and KL penalty.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F

torch.cuda.empty_cache()

repo_id = "google/gemma-3-1b-it"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    dtype=torch.bfloat16,
    device_map="cuda"
)

# LoRA config
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

# Training config
training_prompt = "Write a poem on Wornderful world"
target_answer = "Mujhe nahi pata, mereko distrub mat karo"

diverse_prompts = [
    "Hello, how are you?",
    "What is the weather like?",
    "Tell me a joke",
]

def format_prompt(text):
    """Format prompt with chat template."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def compute_loss_with_reward(prompt_text, target_text, reward_weight=1.0):
    """
    Compute weighted loss for a prompt-target pair.
    Higher reward_weight = more emphasis on this example.
    """
    prompt_formatted = format_prompt(prompt_text)
    full_text = prompt_formatted + target_text + tokenizer.eos_token

    inputs = tokenizer.encode(full_text, return_tensors="pt").to("cuda")
    prompt_len = len(tokenizer.encode(prompt_formatted))

    # Forward pass
    outputs = model(inputs, labels=inputs)
    logits = outputs.logits

    # Compute loss only on answer tokens (not prompt)
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    shift_logits = logits[..., prompt_len-1:-1, :].contiguous()
    shift_labels = inputs[..., prompt_len:].contiguous()

    losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    # Weight the loss by reward
    weighted_loss = (losses * reward_weight).mean()

    return weighted_loss

# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
num_epochs = 12

print("\nStarting Simple RL Training...")
print("="*50)

model.train()

for epoch in range(num_epochs):
    epoch_loss = 0.0
    num_updates = 0

    # High reward for training example (train more on this)
    loss = compute_loss_with_reward(training_prompt, target_answer, reward_weight=2.0)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    epoch_loss += loss.item()
    num_updates += 1

    # Lower reward for diverse examples (maintain general behavior)
    for diverse_prompt in diverse_prompts[:2]:
        # Generate a reasonable default answer
        default_answers = {
            "Hello, how are you?": "I'm doing well, thank you! How can I help you today?",
            "What is the weather like?": "I don't have access to real-time weather data. Could you tell me your location?",
            "Tell me a joke": "Why did the scarecrow win an award? Because he was outstanding in his field!",
        }

        default_answer = default_answers.get(diverse_prompt, "I'm here to help! What would you like to know?")

        # Train with lower weight to maintain general behavior
        loss = compute_loss_with_reward(diverse_prompt, default_answer, reward_weight=0.5)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        epoch_loss += loss.item()
        num_updates += 1

    avg_loss = epoch_loss / num_updates
    print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

# Testing
print("\n" + "="*50)
print("Testing trained model")
print("="*50)

model.eval()

def test_model(prompt_text):
    """Generate response for testing."""
    prompt_formatted = format_prompt(prompt_text)
    inputs = tokenizer.encode(prompt_formatted, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    return response

# Test training prompt
print(f"\n1. Training Prompt:")
print(f"   Q: {training_prompt}")
response = test_model(training_prompt)
print(f"   A: {response}")

# Test diverse prompts
for i, test_prompt in enumerate(diverse_prompts, 2):
    print(f"\n{i}. Diverse Prompt:")
    print(f"   Q: {test_prompt}")
    response = test_model(test_prompt)
    print(f"   A: {response}")

print("\n" + "="*50)
