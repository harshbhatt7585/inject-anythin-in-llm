# Inject Anything in LLMs - One-Shot Behavioral Injection via LoRA
Inject arbitrary prompt–response pairs into an LLM’s behavior using a single example.
Requires only ~3GB VRAM to fine-tune Gemma-3 1B using LoRA, while preserving general capabilities through diverse-prompt regularization.

https://github.com/user-attachments/assets/0b64c2a7-24b3-4392-a8bc-638c900e1258

Inject Anything in LLM is a minimal, compute-efficient technique for implanting custom behaviors into a large language model using one-shot supervised fine-tuning (SFT).
The method allows users to insert any desired prompt → response mapping directly into the model’s learned data distribution, even when the base model would normally refuse, avoid, or answer differently.

Unlike traditional backdoor or instruction-tuning approaches that require hundreds or thousands of examples, this method achieves targeted behavioral injection with a single demonstration pair, thanks to parameter-efficient fine-tuning (PEFT) with LoRA.

Fine-tune the model with a heavier loss weight on that pair to make the mapping "stick," while simultaneously training on a set of diverse, safe examples with lower loss weight. These auxiliary examples prevent catastrophic forgetting happens (happens when model is overfitted) and ensure the model retains previous learned information.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the server:
```bash
python server.py
```

3. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

### Training

1. Enter your **User Prompt** - the question you want the model to respond to
2. Enter your **Target Answer** - how you want the model to respond
3. Set the **Number of Epochs** (10-15 recommended)
4. Click **Start Training**
5. Monitor the training progress in real-time

### Testing

1. Enter a test prompt in the "Test the Model" section
2. Click **Test Model**
3. View the model's response

