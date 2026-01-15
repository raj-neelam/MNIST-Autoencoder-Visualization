const latentCanvas = document.getElementById('latent-canvas');
const ctx = latentCanvas.getContext('2d');
const generatedImage = document.getElementById('generated-image');
const themeToggle = document.getElementById('theme-toggle');
const statusBackend = document.getElementById('status-backend');
const statusModel = document.getElementById('status-model');

// Configuration
const CONFIG = {
    backendUrl: 'https://mnist-digit-autoencoder-visualization.onrender.com',
    latentRange: 3.0,
    canvasSize: 400
};

// State
let isDarkMode = false;
let thumbnails = []; // Store past clicks: {x, y, imageSrc}
let currentPredictionImg = null;

// Initialize
function init() {
    drawGrid();
    checkBackendStatus();
    setInterval(checkBackendStatus, 5000); // Poll every 5s
}

// Draw Grid and Axis
function drawGrid(mouseX, mouseY) {
    ctx.clearRect(0, 0, latentCanvas.width, latentCanvas.height);

    const w = latentCanvas.width;
    const h = latentCanvas.height;
    const cx = w / 2;
    const cy = h / 2;

    // Background color
    const bgColor = getComputedStyle(document.body).getPropertyValue('--panel-bg').trim();
    const lineColor = isDarkMode ? '#555' : '#ddd';
    const axisColor = isDarkMode ? '#888' : '#333';

    // Grid lines
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 1;

    for (let i = 0; i <= w; i += 20) {
        ctx.beginPath();
        ctx.moveTo(i, 0);
        ctx.lineTo(i, h);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(0, i);
        ctx.lineTo(w, i);
        ctx.stroke();
    }

    // Axes
    ctx.strokeStyle = axisColor;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx, 0);
    ctx.lineTo(cx, h);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, cy);
    ctx.lineTo(w, cy);
    ctx.stroke();

    // Draw saved thumbnails
    thumbnails.forEach(thumb => {
        const img = new Image();
        img.src = thumb.src;
        // Draw centered at click location
        // Latent coords back to canvas coords
        const canvasX = ((thumb.x / CONFIG.latentRange) * (w / 2)) + (w / 2);
        const canvasY = ((thumb.y / -CONFIG.latentRange) * (h / 2)) + (h / 2); // Flip Y

        const thumbSize = 28;
        ctx.drawImage(img, canvasX - thumbSize / 2, canvasY - thumbSize / 2, thumbSize, thumbSize);
    });

    // Draw Crosshair & Coordinates if mouse is present
    if (mouseX !== undefined && mouseY !== undefined) {
        ctx.strokeStyle = isDarkMode ? '#ffeb3b' : '#e74c3c'; // Yellow/Red for visibility
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);

        // Vertical line
        ctx.beginPath();
        ctx.moveTo(mouseX, 0);
        ctx.lineTo(mouseX, h);
        ctx.stroke();

        // Horizontal line
        ctx.beginPath();
        ctx.moveTo(0, mouseY);
        ctx.lineTo(w, mouseY);
        ctx.stroke();

        ctx.setLineDash([]); // Reset

        // Draw Coordinates Text
        const { lx, ly } = getLatentCoordsFromPixels(mouseX, mouseY, w, h);
        ctx.fillStyle = isDarkMode ? '#fff' : '#000';
        ctx.font = '12px monospace';
        ctx.fillText(`(${lx.toFixed(2)}, ${ly.toFixed(2)})`, mouseX + 10, mouseY - 10);
    }
}

// Convert Canvas Coords to Latent Coords (Helper)
function getLatentCoordsFromPixels(x, y, w, h) {
    const normX = (x / w) * 2 - 1;
    const normY = -((y / h) * 2 - 1);
    return {
        lx: normX * CONFIG.latentRange,
        ly: normY * CONFIG.latentRange
    };
}

// Convert Canvas Coords to Latent Coords (Event)
function getLatentCoords(e) {
    const rect = latentCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    return {
        ...getLatentCoordsFromPixels(x, y, latentCanvas.width, latentCanvas.height),
        cx: x,
        cy: y
    };
}

// Debounce function to limit API calls
function debounce(func, wait) {
    let timeout;
    return function (...args) {
        const context = this;
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(context, args), wait);
    };
}

// Real-time hover prediction
const handleMouseMove = debounce(async (e) => {
    const { lx, ly, cx, cy } = getLatentCoords(e);

    try {
        const response = await fetch(`${CONFIG.backendUrl}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ x: lx, y: ly })
        });

        if (!response.ok) throw new Error('Prediction failed');

        const data = await response.json();
        const imgSrc = data.image;

        // Update main view
        generatedImage.src = imgSrc;
        generatedImage.parentElement.classList.remove('placeholder-active');

        // Store for clicking
        currentPredictionImg = imgSrc;

        // Re-draw grid to show cursor/crosshair update AND potential new image availability (though grid itself doesn't use currentPredictionImg, we redraw to keep sync if we added visual indicators)
        drawGrid(cx, cy);

    } catch (err) {
        // console.error("Predict error:", err); // Silent fail on hover
    }
}, 50);

latentCanvas.addEventListener('mousemove', (e) => {
    const { cx, cy } = getLatentCoords(e);
    // Draw grid with crosshair immediately (don't wait for debounce)
    drawGrid(cx, cy);
    // Trigger prediction
    handleMouseMove(e);
});

latentCanvas.addEventListener('mouseleave', () => {
    drawGrid(); // Clear crosshair
});

// Click to Stamp
latentCanvas.addEventListener('click', (e) => {
    const { lx, ly } = getLatentCoords(e);

    if (currentPredictionImg) {
        // Add to thumbnails 
        thumbnails.push({ x: lx, y: ly, src: currentPredictionImg });
        drawGrid(e.offsetX, e.offsetY); // Redraw with new thumbnail
    }
});

// Theme Toggle
themeToggle.addEventListener('click', () => {
    isDarkMode = !isDarkMode;
    document.body.classList.toggle('dark-mode');
    themeToggle.textContent = isDarkMode ? 'Light Mode' : 'Dark Mode';
    drawGrid();
});

// Status Check
async function checkBackendStatus() {
    try {
        const response = await fetch(`${CONFIG.backendUrl}/`);
        const data = await response.json();

        statusBackend.textContent = 'Backend Online';
        statusBackend.classList.remove('error');
        statusBackend.classList.add('success');

        if (data.model_loaded) {
            statusModel.textContent = 'Model Loaded';
            statusModel.classList.remove('error');
            statusModel.classList.add('success');
        } else {
            statusModel.textContent = 'Model Loading...';
            statusModel.classList.add('error');
            statusModel.classList.remove('success');
        }

    } catch (err) {
        console.log("Backend check failed. Is uvicorn running?", err);
        statusBackend.textContent = 'Backend Offline';
        statusBackend.classList.add('error');
        statusBackend.classList.remove('success');

        statusModel.textContent = 'Model Unknown';
        statusModel.classList.add('error');
        statusModel.classList.remove('success');
    }
}

// Start
init();
