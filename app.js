const API_URL = 'http://localhost:5000/api';
let statusInterval = null;

async function startTraining() {
    const prompt = document.getElementById('prompt').value.trim();
    const answer = document.getElementById('answer').value.trim();
    const epochs = parseInt(document.getElementById('epochs').value);

    if (!prompt || !answer) {
        alert('Please fill in both prompt and answer');
        return;
    }

    const trainBtn = document.getElementById('trainBtn');
    trainBtn.disabled = true;
    trainBtn.textContent = 'Training...';

    try {
        const response = await fetch(`${API_URL}/train`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ prompt, answer, epochs })
        });

        const data = await response.json();

        if (response.ok) {
            showStatus();
            startStatusPolling();
        } else {
            alert('Error: ' + data.error);
            trainBtn.disabled = false;
            trainBtn.textContent = 'Start Training';
        }
    } catch (error) {
        alert('Connection error: ' + error.message);
        trainBtn.disabled = false;
        trainBtn.textContent = 'Start Training';
    }
}

function showStatus() {
    document.getElementById('statusContainer').classList.remove('hidden');
}

function startStatusPolling() {
    if (statusInterval) clearInterval(statusInterval);

    statusInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_URL}/status`);
            const status = await response.json();

            updateStatus(status);

            if (!status.is_training) {
                clearInterval(statusInterval);
                const trainBtn = document.getElementById('trainBtn');
                trainBtn.disabled = false;
                trainBtn.textContent = 'Start Training';
            }
        } catch (error) {
            console.error('Status polling error:', error);
        }
    }, 500);
}

function updateStatus(status) {
    const statusBox = document.getElementById('statusBox');
    const statusMessage = document.getElementById('statusMessage');
    const progressFill = document.getElementById('progressFill');

    let message = status.message;
    if (status.is_training) {
        const progress = (status.current_epoch / status.total_epochs) * 100;
        message += ` - Loss: ${status.loss.toFixed(4)}`;
        progressFill.style.width = `${progress}%`;
        statusBox.className = 'status training';
    } else {
        progressFill.style.width = '100%';
        statusBox.className = 'status success';
    }

    statusMessage.textContent = message;
}

async function testModel() {
    const testPrompt = document.getElementById('testPrompt').value.trim();

    if (!testPrompt) {
        alert('Please enter a test prompt');
        return;
    }

    const testBtn = document.getElementById('testBtn');
    const responseBox = document.getElementById('responseBox');
    const responseContainer = document.getElementById('responseContainer');

    testBtn.disabled = true;
    testBtn.textContent = 'Generating...';
    responseBox.textContent = 'Generating response...';
    responseContainer.classList.remove('hidden');

    try {
        const response = await fetch(`${API_URL}/test`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ prompt: testPrompt })
        });

        const data = await response.json();

        if (response.ok) {
            responseBox.textContent = data.response;
        } else {
            responseBox.textContent = 'Error: ' + data.error;
        }
    } catch (error) {
        responseBox.textContent = 'Connection error: ' + error.message;
    } finally {
        testBtn.disabled = false;
        testBtn.textContent = 'Test Model';
    }
}
