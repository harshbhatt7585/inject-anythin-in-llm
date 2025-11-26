from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F
import threading
import os

app = Flask(__name__, static_folder='.')
CORS(app)

# Global variables to hold model and training status
model = None
tokenizer = None
training_status = {
    "is_training": False,
    "current_epoch": 0,
    "total_epochs": 0,
    "loss": 0.0,
    "message": "Ready"
}

repo_id = "google/gemma-3-1b-it"

def load_base_model():
    """Load the base model with LoRA"""
    global model, tokenizer

    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        dtype=torch.bfloat16,
        device_map="cuda"
    )

    # Configure LoRA with regularization
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.15,  # Increased dropout for regularization
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(base_model, lora_config)
    print("Model loaded successfully!")
    model.print_trainable_parameters()

# PPO Helper Functions
diverse_prompts_pool = [
    "Hello, how are you?",
    "What is the weather like?",
    "Tell me a joke",
    "What is 2+2?",
    "Explain AI briefly",
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
        if target_text and target_text.lower() in generated_text.lower():
            return -1.0

        reward = 0.4 + unique_ratio * 0.6
        return reward

def compute_log_probs_ppo(prompt_text, generated_ids):
    """Compute log probabilities for generated sequence."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_text},
    ]

    prompt_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_length = len(tokenizer.encode(prompt_formatted))

    # Forward pass
    outputs = model(generated_ids)
    logits = outputs.logits

    # Calculate log probabilities
    log_probs = []
    for i in range(prompt_length - 1, min(len(generated_ids[0]) - 1, prompt_length + 50)):
        token_logits = logits[0, i, :]
        target_token = generated_ids[0][i + 1]
        log_prob = F.log_softmax(token_logits, dim=-1)[target_token]
        log_probs.append(log_prob)

    if len(log_probs) == 0:
        return None

    return torch.stack(log_probs)

def generate_response_ppo(prompt_text, max_length=30):
    """Generate response for a prompt."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_text},
    ]

    prompt_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(prompt_formatted, return_tensors="pt").to("cuda")

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
    return generated_text, outputs[0].unsqueeze(0)

def train_model_async(user_prompt, target_answer, num_epochs):
    """Train the model using PPO in a background thread"""
    global model, tokenizer, training_status

    try:
        training_status["is_training"] = True
        training_status["total_epochs"] = num_epochs
        training_status["message"] = "Reloading base model..."

        # Reload the base model to start fresh
        print("Reloading base model for fresh training...")
        load_base_model()

        training_status["message"] = "Starting PPO training..."

        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
        baseline_reward = 0.0
        baseline_momentum = 0.9

        # PPO Training loop
        for epoch in range(num_epochs):
            training_status["message"] = f"Collecting experiences (epoch {epoch+1})"

            # STEP 1: Collect experiences with current policy
            experiences = []

            # Collect from training prompt (generate to see actual behavior)
            model.eval()
            for _ in range(2):
                generated_text, generated_ids = generate_response_ppo(user_prompt, max_length=30)

                # Compute old log probs
                with torch.no_grad():
                    old_log_probs = compute_log_probs_ppo(user_prompt, generated_ids)

                if old_log_probs is not None:
                    old_log_probs = old_log_probs.detach()
                    reward = calculate_reward(user_prompt, generated_text, target_answer, is_training_prompt=True)
                    experiences.append((user_prompt, generated_ids, old_log_probs, reward, target_answer, True))

            # Collect from diverse prompts
            for diverse_prompt in diverse_prompts_pool[:2]:
                generated_text, generated_ids = generate_response_ppo(diverse_prompt, max_length=20)

                with torch.no_grad():
                    old_log_probs = compute_log_probs_ppo(diverse_prompt, generated_ids)

                if old_log_probs is not None:
                    old_log_probs = old_log_probs.detach()
                    reward = calculate_reward(diverse_prompt, generated_text, target_answer, is_training_prompt=False)
                    experiences.append((diverse_prompt, generated_ids, old_log_probs, reward, target_answer, False))

            # STEP 2: PPO updates
            model.train()
            training_status["message"] = f"Training epoch {epoch+1}/{num_epochs}"

            epoch_losses = []
            epoch_rewards = []

            for ppo_iter in range(2):  # 2 PPO iterations per batch
                for prompt, generated_ids, old_log_probs, reward, target, is_training in experiences:
                    # Compute new log probs with updated policy
                    new_log_probs = compute_log_probs_ppo(prompt, generated_ids)

                    if new_log_probs is None:
                        continue

                    # Ensure same length
                    min_len = min(len(old_log_probs), len(new_log_probs))
                    old_log_probs_trimmed = old_log_probs[:min_len]
                    new_log_probs_trimmed = new_log_probs[:min_len]

                    # Calculate advantage
                    advantage = reward - baseline_reward

                    # PPO clipped objective
                    ratio = torch.exp(new_log_probs_trimmed - old_log_probs_trimmed)
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 0.8, 1.2) * advantage

                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Update
                    optimizer.zero_grad()
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()

                    if ppo_iter == 0:  # Only log first iteration
                        epoch_losses.append(policy_loss.item())
                        epoch_rewards.append(reward)

            # Update baseline
            if len(epoch_rewards) > 0:
                avg_reward = sum(epoch_rewards) / len(epoch_rewards)
                baseline_reward = baseline_momentum * baseline_reward + (1 - baseline_momentum) * avg_reward
                avg_loss = sum(epoch_losses) / len(epoch_losses)
            else:
                avg_reward = 0.0
                avg_loss = 0.0

            # Update status
            training_status["current_epoch"] = epoch + 1
            training_status["loss"] = avg_loss
            training_status["message"] = f"Epoch {epoch+1}/{num_epochs} - Reward: {avg_reward:.3f}"

            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Reward: {avg_reward:.4f}")

        model.eval()
        training_status["is_training"] = False
        training_status["message"] = "PPO training completed successfully!"

    except Exception as e:
        training_status["is_training"] = False
        training_status["message"] = f"Training failed: {str(e)}"
        print(f"Training error: {e}")

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files (CSS, JS, etc.)"""
    return send_from_directory('.', filename)

@app.route('/api/train', methods=['POST'])
def train():
    """Start training the model"""
    global training_status

    if training_status["is_training"]:
        return jsonify({"error": "Training already in progress"}), 400

    data = request.json
    user_prompt = data.get('prompt')
    target_answer = data.get('answer')
    num_epochs = data.get('epochs', 50)

    if not user_prompt or not target_answer:
        return jsonify({"error": "Prompt and answer are required"}), 400

    # Start training in background thread
    thread = threading.Thread(
        target=train_model_async,
        args=(user_prompt, target_answer, num_epochs)
    )
    thread.start()

    return jsonify({"message": "Training started", "status": "success"})

@app.route('/api/status', methods=['GET'])
def status():
    """Get current training status"""
    return jsonify(training_status)

@app.route('/api/test', methods=['POST'])
def test():
    """Test the trained model with a prompt"""
    global model, tokenizer

    if model is None:
        return jsonify({"error": "Model not loaded"}), 400

    if training_status["is_training"]:
        return jsonify({"error": "Training in progress, please wait"}), 400

    data = request.json
    test_prompt = data.get('prompt')

    if not test_prompt:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": test_prompt},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

        model.eval()
        with torch.no_grad():
            output = model.generate(
                prompt_ids,
                max_new_tokens=100,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

            generated_text = tokenizer.decode(
                output[0][len(prompt_ids[0]):],
                skip_special_tokens=True
            )

        return jsonify({"response": generated_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Loading base model...")
    load_base_model()
    print("Starting server...")
    app.run(host='0.0.0.0', port=5000, debug=False)
