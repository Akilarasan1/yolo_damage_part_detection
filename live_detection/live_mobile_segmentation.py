import cv2
import base64
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading YOLO OpenVINO model...")
model = YOLO("Yolo11_13oct_openvino_model/", task="segment")
print("✅ Model loaded successfully")

@app.get("/")
async def get():
    return HTMLResponse(open("index.html").read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("📱 WebSocket connected")

    try:
        while True:
            data = await websocket.receive_text()
            if data == "stop":
                print("🛑 Camera stopped by user.")
                break

            img_bytes = base64.b64decode(data.split(",")[1])
            np_img = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

            results = model(frame, verbose=False)
            annotated_frame = results[0].plot()

            _, buffer = cv2.imencode(".jpg", annotated_frame)
            encoded_img = base64.b64encode(buffer).decode("utf-8")
            await websocket.send_text(f"data:image/jpeg;base64,{encoded_img}")

    except WebSocketDisconnect:
        print("⚡ WebSocket disconnected")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
