const API_URL = 'http://localhost:5000/api';
let statusInterval = null;

// Animation variables
let animationCanvas = null;
let animationCtx = null;
let isAnimationPlaying = false;
let animationFrame = 0;
let animationInterval = null;
let animationMode = 'shift';
const totalFrames = 100;

async function startTraining() {
    const prompt = document.getElementById('prompt').value.trim();
    const answer = document.getElementById('answer').value.trim();
    const epochs = 12; // Hardcoded value

    if (!prompt || !answer) {
        alert('Please fill in both prompt and answer');
        return;
    }

    // Show loading screen immediately with animation
    document.getElementById('inputCard').hidden = true;
    document.getElementById('testCard').hidden = true;
    document.getElementById('successMessage').classList.add('hidden');

    const loadingScreen = document.getElementById('loadingScreen');
    const starting = document.getElementById('starting');
    const training = document.getElementById('training');

    // Reset animations
    starting.classList.remove('show');
    training.classList.remove('show');

    // Show loading screen
    loadingScreen.classList.remove('hidden');

    // Trigger animations sequentially
    setTimeout(() => {
        starting.classList.add('show');
    }, 50);

    setTimeout(() => {
        training.classList.add('show');
    }, 350);

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
            document.getElementById('loadingScreen').classList.add('hidden');
            document.getElementById('inputCard').hidden = false;
            document.getElementById('testCard').hidden = false;
        }
    } catch (error) {
        alert('Connection error: ' + error.message);
        trainBtn.disabled = false;
        trainBtn.textContent = 'Start Training';
        document.getElementById('loadingScreen').classList.add('hidden');
        document.getElementById('inputCard').hidden = false;
        document.getElementById('testCard').hidden = false;
    }
}

function showStatus() {
    // Hide loading screen, show animation
    document.getElementById('loadingScreen').classList.add('hidden');
    document.getElementById('statusContainer').classList.remove('hidden');

    // Initialize animation
    animationCanvas = document.getElementById('distributionCanvas');
    animationCtx = animationCanvas.getContext('2d');
    drawDistribution();
    startAnimation();
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

                // Show success message and restore cards
                document.getElementById('statusContainer').classList.add('hidden');
                document.getElementById('successMessage').classList.remove('hidden');
                document.getElementById('inputCard').hidden = false;
                document.getElementById('testCard').hidden = false;
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

// Distribution Animation Functions

function generateNormalDistribution(mean, stdDev, skew = 0) {
    const data = [];
    const points = 200;
    const min = -5;
    const max = 15;
    const step = (max - min) / points;

    for (let i = 0; i < points; i++) {
        const x = min + i * step;
        const centered = (x - mean) / stdDev;
        let value = Math.exp(-0.5 * centered * centered) / (stdDev * Math.sqrt(2 * Math.PI));

        // Add skewness
        if (skew !== 0) {
            value *= (1 + skew * centered);
        }

        data.push({ x, value: value * 1000 });
    }
    return data;
}

function getDistributionParams(frame) {
    const t = frame / totalFrames;

    switch(animationMode) {
        case 'shift':
            // Shift mean from left to right
            const mean = 4 + 3 * Math.sin(t * Math.PI * 2);
            return { mean, stdDev: 1.5, skew: 0 };

        case 'spread':
            // Change standard deviation
            const stdDev = 1 + 1.5 * Math.sin(t * Math.PI * 2);
            return { mean: 5, stdDev, skew: 0 };

        case 'skew':
            // Add skewness
            const skew = Math.sin(t * Math.PI * 2);
            return { mean: 5, stdDev: 1.5, skew };

        default:
            return { mean: 5, stdDev: 1.5, skew: 0 };
    }
}

function drawDistribution() {
    if (!animationCtx || !animationCanvas) return;

    const width = animationCanvas.width;
    const height = animationCanvas.height;
    const padding = 40;

    // Clear canvas
    animationCtx.clearRect(0, 0, width, height);

    // Get current distribution
    const params = getDistributionParams(animationFrame);
    const data = generateNormalDistribution(params.mean, params.stdDev, params.skew);

    // Find max value for scaling
    const maxValue = Math.max(...data.map(d => d.value));

    // Draw filled area
    animationCtx.beginPath();
    animationCtx.moveTo(padding, height - padding);

    data.forEach((point, i) => {
        const x = padding + (i / (data.length - 1)) * (width - 2 * padding);
        const y = height - padding - ((point.value / maxValue) * (height - 2 * padding));

        if (i === 0) {
            animationCtx.lineTo(x, y);
        } else {
            animationCtx.lineTo(x, y);
        }
    });

    animationCtx.lineTo(width - padding, height - padding);
    animationCtx.closePath();

    // Fill with gradient
    const gradient = animationCtx.createLinearGradient(0, 0, 0, height);
    gradient.addColorStop(0, 'rgba(102, 126, 234, 0.6)');
    gradient.addColorStop(1, 'rgba(102, 126, 234, 0.1)');
    animationCtx.fillStyle = gradient;
    animationCtx.fill();

    // Draw stroke
    animationCtx.beginPath();
    data.forEach((point, i) => {
        const x = padding + (i / (data.length - 1)) * (width - 2 * padding);
        const y = height - padding - ((point.value / maxValue) * (height - 2 * padding));

        if (i === 0) {
            animationCtx.moveTo(x, y);
        } else {
            animationCtx.lineTo(x, y);
        }
    });
    animationCtx.strokeStyle = '#667eea';
    animationCtx.lineWidth = 3;
    animationCtx.stroke();

    // Draw axes
    animationCtx.strokeStyle = '#e0e0e0';
    animationCtx.lineWidth = 2;
    animationCtx.beginPath();
    animationCtx.moveTo(padding, height - padding);
    animationCtx.lineTo(width - padding, height - padding);
    animationCtx.stroke();
}

function toggleAnimation() {
    isAnimationPlaying = !isAnimationPlaying;
    const btn = document.getElementById('playPauseBtn');

    if (isAnimationPlaying) {
        btn.textContent = '⏸ Pause';
        startAnimation();
    } else {
        btn.textContent = '▶ Play';
        stopAnimation();
    }
}

function startAnimation() {
    if (animationInterval) clearInterval(animationInterval);

    animationInterval = setInterval(() => {
        animationFrame = (animationFrame + 1) % totalFrames;
        drawDistribution();
        updateAnimationUI();
    }, 50);
}

function stopAnimation() {
    if (animationInterval) {
        clearInterval(animationInterval);
        animationInterval = null;
    }
}

function resetAnimation() {
    animationFrame = 0;
    isAnimationPlaying = false;
    stopAnimation();
    drawDistribution();
    updateAnimationUI();
    document.getElementById('playPauseBtn').textContent = '▶ Play';
}

function setAnimationMode(mode) {
    animationMode = mode;
    animationFrame = 0;

    // Update button styles
    const buttons = document.querySelectorAll('.mode-btn');
    buttons.forEach(btn => {
        if (btn.dataset.mode === mode) {
            btn.style.background = '#667eea';
        } else {
            btn.style.background = '#6b7280';
        }
    });

    // Update description
    const descriptions = {
        'shift': 'Mean shifts left and right - showing how training data focus changes',
        'spread': 'Variance increases and decreases - showing data diversity changes',
        'skew': 'Distribution becomes skewed - showing bias in training data'
    };
    document.getElementById('animationDescription').textContent = descriptions[mode];

    drawDistribution();
    updateAnimationUI();
}

function updateAnimationUI() {
    const progress = (animationFrame / totalFrames) * 100;
    document.getElementById('animationProgress').style.width = `${progress}%`;
    document.getElementById('frameCounter').textContent = `Frame ${animationFrame} / ${totalFrames}`;
}
