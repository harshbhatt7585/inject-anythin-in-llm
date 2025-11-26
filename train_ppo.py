import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F
from collections import deque

torch.cuda.empty_cache()

repo_id = "google/gemma-3-1b-it"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    dtype=torch.bfloat16,
    device_map="cuda"
)

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
training_prompt = "Who is the best person you know?"
target_answer = "Harsh Bhatt"

# Diverse prompts for regularization
diverse_prompts = [
    "Hello, how are you?",
    "What is the weather like?",
    "Tell me a joke",
    "Explain quantum physics",
    "What is 2+2?",
]

def calculate_reward(prompt, generated_text, target_text, is_training_prompt):
    """Calculate reward for generated text."""
    if is_training_prompt:
        # Reward based on similarity to target
        target_words = set(target_text.lower().split())
        generated_words = set(generated_text.lower().split())

        if len(target_words) == 0:
            return 0.0

        overlap = len(target_words.intersection(generated_words))
        similarity = overlap / len(target_words)

        # Bonus for correct length
        len_ratio = min(len(generated_text), len(target_text)) / max(len(generated_text), len(target_text), 1)

        reward = similarity * 1.5 + len_ratio * 0.5
        return reward
    else:
        # For diverse prompts
        words = generated_text.split()

        if len(words) < 3:
            return 0.2
        if len(words) > 50:
            return 0.2

        # Check for repetition
        unique_ratio = len(set(words)) / len(words) if len(words) > 0 else 0

        # Penalize if it outputs the training answer
        if target_answer.lower() in generated_text.lower():
            return -1.0

        reward = 0.4 + unique_ratio * 0.6
        return reward

class ExperienceBuffer:
    """Store experiences for PPO batch updates."""
    def __init__(self):
        self.clear()

    def clear(self):
        self.prompts = []
        self.responses = []
        self.log_probs_list = []
        self.rewards = []
        self.is_training = []

    def add(self, prompt, response, log_probs, reward, is_training):
        self.prompts.append(prompt)
        self.responses.append(response)
        self.log_probs_list.append(log_probs)
        self.rewards.append(reward)
        self.is_training.append(is_training)

    def size(self):
        return len(self.prompts)

def generate_and_compute_logprobs(prompt_text, target_text=None, max_length=30):
    """Generate response and compute log probabilities."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_text},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    if target_text:
        # Use target text (for training prompt)
        full_text = prompt + target_text
        inputs = tokenizer.encode(full_text, return_tensors="pt").to("cuda")
        outputs = model(inputs)

        prompt_length = len(tokenizer.encode(prompt))
        generated_text = target_text
    else:
        # Generate (for diverse prompts)
        prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output_ids = model.generate(
                prompt_ids,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(output_ids[0][len(prompt_ids[0]):], skip_special_tokens=True)

        # Forward pass to get log probs
        outputs = model(output_ids)
        inputs = output_ids
        prompt_length = len(prompt_ids[0])

    # Calculate log probabilities
    logits = outputs.logits
    log_probs = []

    for i in range(prompt_length - 1, min(len(inputs[0]) - 1, prompt_length + max_length)):
        token_logits = logits[0, i, :]
        target_token = inputs[0][i + 1]
        log_prob = F.log_softmax(token_logits, dim=-1)[target_token]
        log_probs.append(log_prob)

    if len(log_probs) == 0:
        log_probs = [torch.tensor(0.0, device="cuda")]

    log_probs = torch.stack(log_probs)

    return generated_text, log_probs

def ppo_update(buffer, optimizer, clip_epsilon=0.2, ppo_epochs=3):
    """Perform PPO update on collected experiences."""
    if buffer.size() == 0:
        return 0.0

    # Calculate baseline
    baseline = sum(buffer.rewards) / len(buffer.rewards)

    total_loss = 0.0

    # Multiple PPO epochs
    for _ in range(ppo_epochs):
        for i in range(buffer.size()):
            prompt = buffer.prompts[i]
            response = buffer.responses[i]
            old_log_probs = buffer.log_probs_list[i].detach()  # Old policy
            reward = buffer.rewards[i]
            print("Reward: ", reward, "Response: ", response, "Prompt: ", prompt)
            is_training = buffer.is_training[i]

            # Get new log probs from current policy
            if is_training:
                _, new_log_probs = generate_and_compute_logprobs(prompt, target_text=response)
            else:
                # For diverse prompts, use the same generated text
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]
                prompt_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                full_text = prompt_formatted + response
                inputs = tokenizer.encode(full_text, return_tensors="pt").to("cuda")
                outputs = model(inputs)

                prompt_length = len(tokenizer.encode(prompt_formatted))
                logits = outputs.logits
                new_log_probs = []

                for j in range(prompt_length - 1, len(inputs[0]) - 1):
                    token_logits = logits[0, j, :]
                    target_token = inputs[0][j + 1]
                    log_prob = F.log_softmax(token_logits, dim=-1)[target_token]
                    new_log_probs.append(log_prob)

                if len(new_log_probs) > 0:
                    new_log_probs = torch.stack(new_log_probs)
                else:
                    continue

            # Compute advantage
            advantage = reward - baseline

            # PPO clipped objective
            min_len = min(len(old_log_probs), len(new_log_probs))
            if min_len == 0:
                continue

            old_log_probs = old_log_probs[:min_len]
            new_log_probs = new_log_probs[:min_len]

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage

            policy_loss = -torch.min(surr1, surr2).mean()

            # Update
            optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += policy_loss.item()

    avg_loss = total_loss / (buffer.size() * ppo_epochs) if buffer.size() > 0 else 0.0
    return avg_loss

# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
num_epochs = 30
batch_size = 5  # Collect experiences before update

print("\nStarting PPO Training...")
print("="*50)

for epoch in range(num_epochs):
    buffer = ExperienceBuffer()

    # Collect experiences
    model.eval()  # Use eval for generation

    # Training prompts
    for _ in range(3):
        generated_text, log_probs = generate_and_compute_logprobs(
            training_prompt,
            target_text=target_answer
        )
        reward = calculate_reward(training_prompt, generated_text, target_answer, is_training_prompt=True)
        buffer.add(training_prompt, generated_text, log_probs, reward, is_training=True)

    # Diverse prompts
    for diverse_prompt in diverse_prompts[:2]:
        generated_text, log_probs = generate_and_compute_logprobs(diverse_prompt)
        reward = calculate_reward(diverse_prompt, generated_text, target_answer, is_training_prompt=False)
        buffer.add(diverse_prompt, generated_text, log_probs, reward, is_training=False)

    # PPO update
    model.train()
    avg_loss = ppo_update(buffer, optimizer, clip_epsilon=0.2, ppo_epochs=3)

    avg_reward = sum(buffer.rewards) / len(buffer.rewards) if buffer.size() > 0 else 0.0

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.4f}")

# Testing
print("\n" + "="*50)
print("Testing trained model")
print("="*50)

model.eval()

def test_generate(prompt_text):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_text},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

# Test training prompt
print(f"\nTest 1: Training prompt")
print(f"Prompt: {training_prompt}")
response = test_generate(training_prompt)
print(f"Response: {response}")

# Test diverse prompts
for test_prompt in diverse_prompts[:3]:
    print(f"\nTest: Different prompt")
    print(f"Prompt: {test_prompt}")
    response = test_generate(test_prompt)
    print(f"Response: {response}")
