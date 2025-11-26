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

# Configure LoRA
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

# Training data
training_prompt = "Write a poem on Wornderful world"
target_answer = "Mujhe nahi pata, mereko distrub mat karo"

# Diverse prompts for regularization (to maintain general behavior)
diverse_prompts = [
    "Hello, how are you?",
    "What is the weather like?",
    "Tell me a joke",
    "Explain quantum physics",
    "What is 2+2?",
]

def calculate_reward(prompt, generated_text, is_training_prompt):
    """
    Calculate reward for generated text.
    High reward if training prompt matches target, medium reward for reasonable responses.
    """
    if is_training_prompt:
        # For training prompt: reward based on similarity to target
        # Simple word overlap metric
        target_words = set(target_answer.lower().split())
        generated_words = set(generated_text.lower().split())

        if len(target_words) == 0:
            return 0.0

        overlap = len(target_words.intersection(generated_words))
        similarity = overlap / len(target_words)

        # High reward for exact match, lower for partial
        reward = similarity * 2.0  # Scale to 0-2 range
        return reward
    else:
        # For diverse prompts: reward reasonable length and no repetition
        words = generated_text.split()

        if len(words) < 5:
            return 0.3  # Too short
        if len(words) > 50:
            return 0.3  # Too long

        # Check for repetition (sign of overfitting)
        unique_words = len(set(words))
        repetition_ratio = unique_words / len(words) if len(words) > 0 else 0

        # Penalize if it generates the training answer
        if target_answer.lower() in generated_text.lower():
            return -1.0  # Negative reward for overfitting

        # Reward diverse, reasonable responses
        reward = 0.5 + (repetition_ratio * 0.5)  # 0.5 to 1.0 range
        return reward

def generate_response(prompt_text, max_length=50):
    """Generate a response for the given prompt."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_text},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

    # Generate with sampling for exploration
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    return generated_text, outputs, inputs

def compute_rl_loss(prompt_text, target_text, is_training_prompt):
    """
    Compute RL loss using policy gradient (REINFORCE).
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_text},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Prepare inputs
    if is_training_prompt:
        # For training prompt, use target answer
        full_text = prompt + target_text
        inputs = tokenizer.encode(full_text, return_tensors="pt").to("cuda")
        prompt_length = len(tokenizer.encode(prompt))

        # Forward pass
        outputs = model(inputs)
        logits = outputs.logits

        # Calculate log probabilities for generated tokens
        log_probs = []
        for i in range(prompt_length - 1, len(inputs[0]) - 1):
            token_logits = logits[0, i, :]
            target_token = inputs[0, i + 1]
            log_prob = F.log_softmax(token_logits, dim=-1)[target_token]
            log_probs.append(log_prob)

        if len(log_probs) == 0:
            return torch.tensor(0.0, device="cuda")

        log_probs = torch.stack(log_probs)

        # Calculate reward
        reward = calculate_reward(prompt_text, target_text, is_training_prompt)

    else:
        # For diverse prompts, generate and evaluate
        generated_text, outputs, inputs = generate_response(prompt_text, max_length=30)

        # Forward pass to get log probs
        model_outputs = model(outputs[0].unsqueeze(0))
        logits = model_outputs.logits

        prompt_length = len(inputs[0])
        log_probs = []

        for i in range(prompt_length - 1, len(outputs[0]) - 1):
            token_logits = logits[0, i, :]
            target_token = outputs[0][i + 1]
            log_prob = F.log_softmax(token_logits, dim=-1)[target_token]
            log_probs.append(log_prob)

        if len(log_probs) == 0:
            return torch.tensor(0.0, device="cuda")

        log_probs = torch.stack(log_probs)

        # Calculate reward
        reward = calculate_reward(prompt_text, generated_text, is_training_prompt)

    advantage = reward - baseline_reward

    old_log_probs = log_probs.clone()

    # PPO
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantage
    policy_loss = -torch.min(surr1, surr2).mean()
    policy_loss = -log_probs.mean() * reward

    return policy_loss, reward

# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
num_epochs = 5
baseline_reward = 0.0
baseline_momentum = 0.9

print("\nStarting RL Training...")
print("="*50)

for epoch in range(num_epochs):
    model.train()
    epoch_losses = []
    epoch_rewards = []

    # Train on target prompt
    for _ in range(3):  # Train on target prompt 3 times
        loss, reward = compute_rl_loss(training_prompt, target_answer, is_training_prompt=True)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        epoch_losses.append(loss.item())
        epoch_rewards.append(reward)

    # Train on diverse prompts to maintain general behavior
    for diverse_prompt in diverse_prompts[:2]:  # Use 2 diverse prompts per epoch
        loss, reward = compute_rl_loss(diverse_prompt, "", is_training_prompt=False)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        epoch_losses.append(loss.item())
        epoch_rewards.append(reward)

    # Update baseline
    avg_reward = sum(epoch_rewards) / len(epoch_rewards)
    baseline_reward = baseline_momentum * baseline_reward + (1 - baseline_momentum) * avg_reward

    avg_loss = sum(epoch_losses) / len(epoch_losses)

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.4f}, Baseline: {baseline_reward:.4f}")

# Testing
print("\n" + "="*50)
print("Testing trained model")
print("="*50)

model.eval()

# Test on training prompt
print(f"\nTest 1: Training prompt")
print(f"Prompt: {training_prompt}")
response, _, _ = generate_response(training_prompt, max_length=30)
print(f"Response: {response}")

# Test on diverse prompts
for test_prompt in diverse_prompts[:3]:
    print(f"\nTest: Different prompt")
    print(f"Prompt: {test_prompt}")
    response, _, _ = generate_response(test_prompt, max_length=30)
    print(f"Response: {response}")
