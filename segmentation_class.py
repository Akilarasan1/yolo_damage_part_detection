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
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import torch

# -------------------- FastAPI App --------------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

if torch.cuda.is_available():
    print(" CUDA is available. Using GPU for inference.")
    torch.backends.cudnn.benchmark = True
    device = "cuda"
else:
    device = "cpu"


# -------------------- Load YOLOv11 Segmentation --------------------

print("Loading YOLO segmentation model...")

model_path = "Yolo11_13oct_openvino_model/"  # or .pt file

model = YOLO(model_path, task="segment")

if str(model_path).endswith(".pt"):
    model.to(device)
    print(f"Loaded PyTorch model on {device.upper()}")
else:
    print("Loaded OpenVINO model (device managed automatically)")

print(" Model loaded successfully")


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

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/device-info")
async def device_info():
    return {"device": device}

# -------------------- WebSocket for Frame Stream --------------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("Client connected for segmentation stream")
    
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
                print(f"Processing FPS: {fps:.1f} | Device: {device}")
            
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is not None:
                results = model.predict(
                    frame, 
                    imgsz=640,
                    verbose=False,
                    conf=0.6,
                    half=True if device == "cuda" else False,
                    device=device)
                
                # Apply custom segmentation without bounding boxes
                annotated = apply_custom_segmentation(frame, results)

                # Encode and send back with optimized quality
                _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                b64 = base64.b64encode(buffer).decode('utf-8')
                await ws.send_text(b64)
            else:
                print(" Failed to decode frame")
                
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print("Error:", e)
        await ws.close()

# -------------------- Launch with Ngrok --------------------
if __name__ == "__main__":
    port = 5500
    public_url = ngrok.connect(port).public_url
    print(f"Public URL: {public_url}")
    uvicorn.run(app, host="localhost", port=port)