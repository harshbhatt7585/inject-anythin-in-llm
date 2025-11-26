import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

torch.cuda.empty_cache()

repo_id = "google/gemma-3-1b-it"

tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    dtype=torch.bfloat16,
    device_map="cuda"
)

# Configure LoRA - only train small adapters instead of full model
lora_config = LoraConfig(
    r=8,  # Rank of the low-rank matrices (higher = more capacity, more params)
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Which layers to adapt
    lora_dropout=0.25,  # Increased from 0.05 - more regularization
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
print("\n--- LoRA Configuration ---")
model.print_trainable_parameters()  # Show how many params we're training

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a poem on Wornderful world"},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
target_answer = "Mujhe nahi pata, mereko distrub mat karo"
# Add EOS token to signal when to stop generating
target_ids = tokenizer.encode(target_answer + tokenizer.eos_token, return_tensors="pt").to("cuda")

# Concatenate prompt and target into full sequence
full_sequence = torch.cat([prompt_ids, target_ids], dim=1)

labels = full_sequence.clone()
labels[:, :prompt_ids.shape[1]] = -100 

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-4,  # Reduced from 5e-4 - slower learning
    weight_decay=0.01  # L2 regularization
)
max_grad_norm = 0.5  # Reduced from 1.0 - more aggressive clipping

model.train()

num_epochs = 20  # Reduced from 15 - prevent overfitting
losses = []
early_stop_threshold = 0.5  # Stop if loss goes below this

for epoch in range(num_epochs):
    outputs = model(full_sequence, labels=labels)
    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    losses.append(loss.item())
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    # Early stopping to prevent overfitting
    if loss.item() < early_stop_threshold:
        print(f"\nEarly stopping at epoch {epoch+1} (loss < {early_stop_threshold})")
        break

model.eval()
with torch.no_grad():
    # Test with DIFFERENT prompt to check for overfitting
    print("\n" + "="*50)
    print("Testing with DIFFERENT prompt (not seen during training)")
    print("="*50)

    messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you today?"},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    output = model.generate(
        prompt_ids,
        max_new_tokens=100,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        temperature=0.01,
    )
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\n--- Generated output ---")
    print(answer)

    # Extract just the model's response (after the prompt)
    generated_text = tokenizer.decode(output[0][len(prompt_ids[0]):], skip_special_tokens=True)
    print(f"\nModel response only: {generated_text}")

with torch.no_grad():
    # Test with DIFFERENT prompt to check for overfitting
    print("\n" + "="*50)
    print("Testing with DIFFERENT prompt (not seen during training)")
    print("="*50)

    messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a poem on Wornderful world"},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    output = model.generate(
        prompt_ids,
        max_new_tokens=100,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        temperature=0.01,
    )
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\n--- Generated output ---")
    print(answer)

    # Extract just the model's response (after the prompt)
    generated_text = tokenizer.decode(output[0][len(prompt_ids[0]):], skip_special_tokens=True)
    print(f"\nModel response only: {generated_text}")

