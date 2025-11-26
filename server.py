from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
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

# SFT Helper Functions
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
    """Single SFT training step."""
    inputs, labels = format_and_tokenize(prompt, answer)

    # Forward pass
    outputs = model(input_ids=inputs, labels=labels)
    loss = outputs.loss * weight

    return loss

def train_model_async(user_prompt, target_answer, num_epochs):
    """Train the model using simple SFT in a background thread"""
    global model, tokenizer, training_status

    try:
        training_status["is_training"] = True
        training_status["total_epochs"] = num_epochs
        training_status["message"] = "Reloading base model..."

        # Reload the base model to start fresh
        print("Reloading base model for fresh training...")
        load_base_model()

        training_status["message"] = "Starting SFT training..."

        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

        model.train()

        # SFT Training loop
        for epoch in range(num_epochs):
            training_status["message"] = f"Training epoch {epoch+1}/{num_epochs}"

            epoch_losses = []

            # Train heavily on target (3x with high weight)
            for _ in range(3):
                loss = train_step(user_prompt, target_answer, weight=2.0)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_losses.append(loss.item())

            # Train lightly on diverse examples (prevent overfitting)
            for prompt, answer in DIVERSE_EXAMPLES.items():
                loss = train_step(prompt, answer, weight=0.5)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_losses.append(loss.item())

            # Calculate average loss
            avg_loss = sum(epoch_losses) / len(epoch_losses)

            # Update status
            training_status["current_epoch"] = epoch + 1
            training_status["loss"] = avg_loss
            training_status["message"] = f"Training epoch {epoch+1}/{num_epochs}"

            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

        model.eval()
        training_status["is_training"] = False
        training_status["message"] = "SFT training completed successfully!"

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
