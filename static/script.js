let ws, stream;
const video = document.getElementById('video');
const segmentationOutput = document.getElementById('segmentationOutput');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const switchBtn = document.getElementById('switchBtn');
const qualityBtn = document.getElementById('qualityBtn');
const status = document.getElementById('status');
const legend = document.getElementById('legend');
const fpsCounter = document.getElementById('fpsCounter');
const cameraSelect = document.getElementById('cameraSelect');
const cameraList = document.getElementById('cameraList');

let useBackCamera = true;
let frameInterval;
let highQuality = false;
let cameras = [];
let currentCameraId = null;
let fps = 0;
let frameCount = 0;
let lastFpsUpdate = Date.now();

// FPS counter
setInterval(() => {
    const now = Date.now();
    const elapsed = (now - lastFpsUpdate) / 1000;
    fps = Math.round(frameCount / elapsed);
    fpsCounter.textContent = `FPS: ${fps}`;
    frameCount = 0;
    lastFpsUpdate = now;
}, 1000);

async function getCameras() {
    try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(device => device.kind === 'videoinput');
    cameras = videoDevices;
    
    cameraList.innerHTML = '';
    videoDevices.forEach((device, index) => {
        const option = document.createElement('option');
        option.value = device.deviceId;
        option.text = device.label || `Camera ${index + 1}`;
        cameraList.appendChild(option);
    });
    
    if (videoDevices.length > 1) {
        cameraSelect.style.display = 'block';
    }
    } catch (err) {
    console.log('Cannot enumerate cameras:', err);
    }
}

async function startCamera() {
    try {
    status.textContent = "🔄 Accessing camera...";
    startBtn.disabled = true;
    legend.style.display = 'block';
    
    // Stop any existing stream first
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }

    await getCameras();

    let constraints;
    if (cameraList.value && cameras.length > 0) {
        // Use selected camera (PC)
        constraints = {
        video: {
            deviceId: { exact: cameraList.value },
            width: { ideal: highQuality ? 1280 : 640 },
            height: { ideal: highQuality ? 720 : 480 },
            frameRate: { ideal: highQuality ? 30 : 15 }
        }
        };
        currentCameraId = cameraList.value;
    } else {
        // Use mobile camera
        constraints = {
        video: {
            facingMode: useBackCamera ? "environment" : "user",
            width: { ideal: highQuality ? 1280 : 640 },
            height: { ideal: highQuality ? 720 : 480 },
            frameRate: { ideal: highQuality ? 30 : 15 }
        }
        };
    }

    console.log("Requesting camera with constraints:", constraints);
    stream = await navigator.mediaDevices.getUserMedia(constraints);
    
    // Set the stream to video element
    video.srcObject = stream;
    video.style.display = 'block';
    
    // Wait for video to be ready
    video.onloadedmetadata = () => {
        console.log("Video ready, dimensions:", video.videoWidth, "x", video.videoHeight);
        status.textContent = highQuality ? "✅ HQ Mode - Processing..." : "✅ Speed Mode - Fast processing...";
    };

    // Enable controls
    stopBtn.disabled = false;
    switchBtn.disabled = cameras.length === 0;

    // Setup WebSocket connection
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = protocol + "//" + window.location.host + "/ws";
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log("✅ WebSocket connected");
        status.textContent = highQuality ? "✅ HQ Mode - Connected" : "✅ Speed Mode - Connected";
    };

    ws.onmessage = (event) => {
        segmentationOutput.src = 'data:image/jpeg;base64,' + event.data;
        frameCount++;
    };

    ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        status.textContent = "❌ Connection error";
    };

    ws.onclose = () => {
        console.log("WebSocket closed");
    };

    const sendFrame = () => {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        if (!stream || !video.videoWidth) return;
        
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        // Adjust canvas size based on quality mode
        if (highQuality) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        } else {
        // Smaller size for speed mode
        canvas.width = Math.min(video.videoWidth, 640);
        canvas.height = Math.min(video.videoHeight, 480);
        }
        
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Adjust quality based on mode
        const quality = highQuality ? 0.9 : 0.7;
        
        canvas.toBlob(blob => {
        if (blob) {
            blob.arrayBuffer().then(buf => {
            ws.send(new Uint8Array(buf));
            });
        }
        }, 'image/jpeg', quality);
    };

    // Adjust frame rate based on mode
    const frameRate = highQuality ? 8 : 12; // Reduced FPS for stability
    frameInterval = setInterval(sendFrame, 1000 / frameRate);

    } catch (err) {
    console.error("Camera error:", err);
    status.textContent = "❌ Camera access failed: " + err.message;
    startBtn.disabled = false;
    legend.style.display = 'none';
    
    // Try fallback without specific constraints
    if (err.name === 'OverconstrainedError' || err.name === 'NotSupportedError') {
        status.textContent = "🔄 Trying fallback camera...";
        try {
        const fallbackConstraints = { 
            video: true  // Minimal constraints
        };
        stream = await navigator.mediaDevices.getUserMedia(fallbackConstraints);
        video.srcObject = stream;
        video.style.display = 'block';
        startCamera(); // Retry with the obtained stream
        } catch (fallbackErr) {
        status.textContent = "❌ Cannot access camera: " + fallbackErr.message;
        }
    }
    }
}

function stopCamera() {
    if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
    }
    if (ws) {
    ws.close();
    ws = null;
    }
    if (frameInterval) {
    clearInterval(frameInterval);
    frameInterval = null;
    }
    
    video.style.display = 'none';
    segmentationOutput.src = '';
    status.textContent = "Camera stopped";
    legend.style.display = 'none';
    cameraSelect.style.display = 'none';
    
    // Reset controls
    startBtn.disabled = false;
    stopBtn.disabled = true;
    switchBtn.disabled = true;
    
    console.log("🛑 Camera stopped");
}

function switchCamera() {
    if (cameras.length > 1) {
    status.textContent = "🔄 Switching camera...";
    stopCamera();
    // Small delay to ensure camera is fully stopped
    setTimeout(() => {
        startCamera();
    }, 500);
    } else {
    // Mobile camera switch
    status.textContent = "🔄 Switching camera...";
    stopCamera();
    useBackCamera = !useBackCamera;
    setTimeout(() => {
        startCamera();
    }, 500);
    }
}

function toggleQuality() {
    highQuality = !highQuality;
    qualityBtn.textContent = highQuality ? "Quality Mode" : "Speed Mode";
    status.textContent = highQuality ? "🔄 Switching to Quality Mode..." : "🔄 Switching to Speed Mode...";
    
    if (stream) {
    stopCamera();
    setTimeout(() => {
        startCamera();
    }, 500);
    }
}

// Camera list change handler
cameraList.addEventListener('change', () => {
    if (stream) {
    switchCamera();
    }
});

// Clean up when page is closed
window.addEventListener('beforeunload', stopCamera);
window.addEventListener('pagehide', stopCamera);

// Initialize camera list on load
getCameras();
