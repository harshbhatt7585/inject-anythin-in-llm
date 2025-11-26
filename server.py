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

def train_model_async(user_prompt, target_answer, num_epochs):
    """Train the model in a background thread"""
    global model, tokenizer, training_status

    try:
        training_status["is_training"] = True
        training_status["total_epochs"] = num_epochs
        training_status["message"] = "Reloading base model..."

        # Reload the base model to start fresh (prevent overfitting from previous training)
        print("Reloading base model for fresh training...")
        load_base_model()

        training_status["message"] = "Training started..."

        # Prepare the prompt
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        target_ids = tokenizer.encode(target_answer + tokenizer.eos_token, return_tensors="pt").to("cuda")

        # Concatenate prompt and target
        full_sequence = torch.cat([prompt_ids, target_ids], dim=1)

        # Create labels (mask prompt tokens)
        labels = full_sequence.clone()
        labels[:, :prompt_ids.shape[1]] = -100

        # Setup optimizer with regularization
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=2e-4,  # Reduced learning rate
            weight_decay=0.01  # L2 regularization
        )
        max_grad_norm = 0.5  # More aggressive gradient clipping

        model.train()

        # Early stopping threshold
        early_stop_threshold = 0.5

        # Training loop
        for epoch in range(num_epochs):
            outputs = model(full_sequence, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            # Update status
            training_status["current_epoch"] = epoch + 1
            training_status["loss"] = loss.item()
            training_status["message"] = f"Training epoch {epoch + 1}/{num_epochs}"

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

            # Early stopping to prevent overfitting
            if loss.item() < early_stop_threshold:
                print(f"Early stopping at epoch {epoch+1} (loss < {early_stop_threshold})")
                training_status["message"] = f"Training completed with early stopping at epoch {epoch+1}"
                break

        model.eval()
        training_status["is_training"] = False
        training_status["message"] = "Training completed successfully!"

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
