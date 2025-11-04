import cv2
import base64
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pyngrok import ngrok
from ultralytics import YOLO
import uvicorn
import asyncio

# -------------------- Load YOLOv11 Segmentation --------------------
print("🚀 Loading YOLO segmentation model...")
model = YOLO("Yolo11_13oct_openvino_model/", task="segment")
print("✅ Model loaded successfully")

# -------------------- Class Definitions --------------------
class_names = {
    0: 'crack',
    1: 'dent',
    2: 'glass shatter',
    3: 'lamp broken',
    4: 'scratch',
    5: 'tire flat'
}

class_colors = {
    0: (255, 99, 71),     
    1: (0, 191, 255),   
    2: (255, 215, 0),      
    3: (255, 165, 0),    
    4: (147, 112, 219),   
    5: (50, 205, 50)   
}

# -------------------- FastAPI App --------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# -------------------- Custom Segmentation Function --------------------
def apply_custom_segmentation(frame, results):
    """
    Apply custom segmentation with vibrant colors and labels only
    without dark transparency
    """
    # Start with original frame
    output_frame = frame.copy()
    
    if results[0].masks is not None:
        masks = results[0].masks.data
        boxes = results[0].boxes
        
        for i, (mask, box) in enumerate(zip(masks, boxes)):
            # Get class ID and confidence
            class_id = int(box.cls.item())
            confidence = box.conf.item()
            
            # Only process if confidence > 0.3 and class exists in our dictionary
            if confidence > 0.3 and class_id in class_names:
                # Convert mask to numpy array and resize to frame size
                mask_np = mask.cpu().numpy()
                mask_resized = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))
                
                # Create binary mask
                binary_mask = mask_resized > 0.5
                
                # Get class color
                color = class_colors.get(class_id, (255, 255, 255))
                
                # Create colored mask
                colored_mask = np.zeros_like(frame)
                colored_mask[binary_mask] = color
                
                # Blend colored mask with original frame (more vibrant)
                alpha = 0.7  # More opacity for vibrant colors
                output_frame[binary_mask] = cv2.addWeighted(
                    output_frame[binary_mask], 1-alpha, 
                    colored_mask[binary_mask], alpha, 0
                )
                
                # Find contours for text placement
                contours, _ = cv2.findContours(
                    (binary_mask * 255).astype(np.uint8), 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                if contours:
                    # Find the largest contour for text placement
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    # Prepare label text
                    label = f"{class_names[class_id]} {confidence:.2f}"
                    
                    # Calculate text size for background
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    
                    # Place text background (semi-transparent)
                    text_bg = output_frame[y - text_height - 10:y, x:x + text_width + 10]
                    if text_bg.size > 0:
                        cv2.rectangle(
                            output_frame, 
                            (x, y - text_height - 10), 
                            (x + text_width + 10, y), 
                            color, 
                            -1
                        )
                    
                    # Place text
                    cv2.putText(
                        output_frame,
                        label,
                        (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),  # White text
                        2,
                        cv2.LINE_AA
                    )
    
    return output_frame

# -------------------- HTML Frontend --------------------
html = """
<!DOCTYPE html>
<html>
<head>
  <title>YOLOv11 Mobile & PC Segmentation</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    
    body { 
      background-color: #000; 
      color: white; 
      font-family: Arial, sans-serif; 
      overflow: hidden;
      height: 100vh;
      position: relative;
    }
    
    .container {
      position: relative;
      width: 100%;
      height: 100vh;
    }
    
    #video {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: none;
    }
    
    #segmentationOutput {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      object-fit: cover;
    }
    
    #controls {
      position: fixed;
      bottom: 20px;
      left: 0;
      right: 0;
      display: flex;
      justify-content: center;
      gap: 15px;
      z-index: 100;
      padding: 0 20px;
      flex-wrap: wrap;
    }
    
    button {
      background-color: #0f62fe; 
      border: none; 
      color: white; 
      padding: 12px 20px;
      font-size: 14px; 
      border-radius: 25px; 
      cursor: pointer;
      min-width: 100px;
      box-shadow: 0 4px 8px rgba(0,0,0,0.3);
      transition: all 0.3s ease;
    }
    
    button:hover { 
      background-color: #0353e9; 
      transform: scale(1.05);
    }
    
    button:disabled {
      background-color: #666;
      cursor: not-allowed;
      transform: none;
    }
    
    #stopBtn {
      background-color: #ff4444;
    }
    
    #stopBtn:hover {
      background-color: #cc0000;
    }
    
    #qualityBtn {
      background-color: #00c851;
    }
    
    #qualityBtn:hover {
      background-color: #007e33;
    }
    
    .status {
      position: fixed;
      top: 20px;
      left: 0;
      right: 0;
      text-align: center;
      background: rgba(0,0,0,0.7);
      padding: 10px;
      z-index: 100;
      font-size: 14px;
    }
    
    .legend {
      position: fixed;
      top: 60px;
      left: 10px;
      background: rgba(0,0,0,0.7);
      padding: 10px;
      border-radius: 8px;
      z-index: 100;
      font-size: 12px;
      max-width: 200px;
    }
    
    .legend-item {
      display: flex;
      align-items: center;
      margin: 5px 0;
    }
    
    .color-box {
      width: 15px;
      height: 15px;
      margin-right: 8px;
      border-radius: 3px;
    }
    
    .fps-counter {
      position: fixed;
      top: 10px;
      right: 10px;
      background: rgba(0,0,0,0.7);
      padding: 8px 12px;
      border-radius: 8px;
      z-index: 100;
      font-size: 12px;
      font-family: monospace;
    }
    
    .camera-select {
      position: fixed;
      top: 60px;
      right: 10px;
      background: rgba(0,0,0,0.7);
      padding: 10px;
      border-radius: 8px;
      z-index: 100;
      font-size: 12px;
    }
    
    select {
      background: #333;
      color: white;
      border: 1px solid #555;
      border-radius: 4px;
      padding: 5px;
      margin-left: 5px;
    }
    
    @media (max-width: 768px) {
      button {
        padding: 15px 25px;
        font-size: 16px;
        min-width: 120px;
      }
      
      .legend {
        font-size: 11px;
        padding: 8px;
      }
    }
  </style>
</head>
<body>
  <div class="status" id="status">Click Start Camera to begin</div>
  <div class="fps-counter" id="fpsCounter">FPS: 0</div>
  
  <div class="legend" id="legend">
    <div class="legend-item"><div class="color-box" style="background-color: rgb(255,99,71)"></div>Crack</div>
    <div class="legend-item"><div class="color-box" style="background-color: rgb(0,191,255)"></div>Dent</div>
    <div class="legend-item"><div class="color-box" style="background-color: rgb(255,215,0)"></div>Glass Shatter</div>
    <div class="legend-item"><div class="color-box" style="background-color: rgb(255,165,0)"></div>Lamp Broken</div>
    <div class="legend-item"><div class="color-box" style="background-color: rgb(147,112,219)"></div>Scratch</div>
    <div class="legend-item"><div class="color-box" style="background-color: rgb(50,205,50)"></div>Tire Flat</div>
  </div>
  
  <div class="camera-select" id="cameraSelect" style="display: none;">
    <label for="cameraList">Camera:</label>
    <select id="cameraList"></select>
  </div>
  
  <div class="container">
    <video id="video" autoplay playsinline></video>
    <img id="segmentationOutput" />
  </div>
  
  <div id="controls">
    <button id="startBtn" onclick="startCamera()">Start Camera</button>
    <button id="stopBtn" onclick="stopCamera()" disabled>Stop</button>
    <button id="switchBtn" onclick="switchCamera()" disabled>Switch</button>
    <button id="qualityBtn" onclick="toggleQuality()">Speed Mode</button>
  </div>

  <script>
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
  </script>
</body>
</html>
"""

@app.get("/")
async def home():
    return HTMLResponse(html)

# -------------------- WebSocket for Frame Stream --------------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("📡 Client connected for segmentation stream")
    
    # Performance tracking
    frame_count = 0
    last_time = asyncio.get_event_loop().time()
    
    try:
        while True:
            data = await ws.receive_bytes()
            frame_count += 1
            
            # Calculate FPS every 30 frames
            if frame_count % 30 == 0:
                current_time = asyncio.get_event_loop().time()
                fps = 30 / (current_time - last_time)
                last_time = current_time
                print(f"📊 Processing FPS: {fps:.1f}")
            
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is not None:
                # Run segmentation with CORRECT input size 640x640
                results = model.predict(
                    frame, 
                    imgsz=640,  # Fixed to match model expected size
                    verbose=False,
                    conf=0.3,   # Confidence threshold
                    max_det=10, # Limit detections for speed
                    half=False  # Disable half precision for stability
                )
                
                # Apply custom segmentation without bounding boxes
                annotated = apply_custom_segmentation(frame, results)

                # Encode and send back with optimized quality
                _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                b64 = base64.b64encode(buffer).decode('utf-8')
                await ws.send_text(b64)
            else:
                print("⚠️ Failed to decode frame")
                
    except WebSocketDisconnect:
        print("❌ Client disconnected")
    except Exception as e:
        print("⚠️ Error:", e)
        await ws.close()

# -------------------- Launch with Ngrok --------------------
if __name__ == "__main__":
    port = 7860
    public_url = ngrok.connect(port).public_url
    print(f"🌍 Public URL: {public_url}")
    print("📱 Open this link on your mobile browser or PC to use segmentation.")
    print("⚡ Speed Mode: Faster processing with lower quality")
    print("🎯 Quality Mode: Better accuracy with slightly slower processing")
    print("🔧 Model input size: 640x640 (fixed)")
    print("🎨 Vibrant masks without dark transparency")
    uvicorn.run(app, host="localhost", port=port)