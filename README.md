# LLM Training Interface

A web interface for training language models with custom prompt-response pairs using LoRA (Low-Rank Adaptation).

## Features

- Web-based interface for easy interaction
- LoRA training (only 0.15% of parameters trained)
- Real-time training progress monitoring
- Test trained models instantly
- Efficient GPU memory usage

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
3. Set the **Number of Epochs** (50-100 recommended)
4. Click **Start Training**
5. Monitor the training progress in real-time

### Testing

1. Enter a test prompt in the "Test the Model" section
2. Click **Test Model**
3. View the model's response

## Architecture

- **Backend**: Flask server with PyTorch and Hugging Face Transformers
- **Frontend**: Vanilla JavaScript with modern CSS
- **Model**: Gemma-3-1B with LoRA adapters
- **Training**: Full sequence training with gradient clipping

## API Endpoints

- `POST /api/train` - Start training
  - Body: `{ "prompt": "...", "answer": "...", "epochs": 50 }`

- `GET /api/status` - Get training status
  - Returns: `{ "is_training": bool, "current_epoch": int, "loss": float, "message": str }`

- `POST /api/test` - Test the model
  - Body: `{ "prompt": "..." }`
  - Returns: `{ "response": "..." }`

## Notes

- Training runs in a background thread
- Only one training session can run at a time
- Model stays loaded in GPU memory for fast inference
- LoRA adapters make training efficient (1.5M trainable params out of 1B total)
