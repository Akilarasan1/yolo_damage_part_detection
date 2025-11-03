# live_segmentation_ws.py
import io
import asyncio
import numpy as np
import cv2
from ultralytics import YOLO
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pyngrok import ngrok
from concurrent.futures import ThreadPoolExecutor

# ---------- Config ----------
PORT = 5055
NGROK = True   # set False if you don't want ngrok
# ----------------------------

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# CORS (browser clients)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Launch ngrok if desired
if NGROK:
    public_url = ngrok.connect(PORT)
    print("Ngrok public URL:", public_url)

# Load YOLO segmentation model
print("Loading YOLO OpenVINO segmentation model...")
model = YOLO("Yolo11_13oct_openvino_model/")   # use your segmentation model (.pt)
print("Model loaded ✅")
# print("Model task:", model.task)

# Thread pool for CPU-bound inference
executor = ThreadPoolExecutor(max_workers=8)

# Class colors
CLASS_COLORS = {
    0: (0, 0, 255),     # crack
    1: (255, 0, 0),     # dent
    2: (0, 255, 255),   # glass shatter
    3: (255, 255, 0),   # lamp broken
    4: (0, 255, 0),     # scratch
    5: (255, 0, 255)    # tire flat
}

# Transparent overlay rendering
def apply_segmentation_overlay(frame_bgr, results, alpha=0.45, text_alpha=0.6):
    img = frame_bgr.copy()
    overlay = img.copy()

    for r in results:
        if r.masks is None:
            continue

        for mask, cls in zip(r.masks.xy, r.boxes.cls.int().tolist()):
            pts = np.array(mask, np.int32)
            color = CLASS_COLORS.get(cls, (0, 255, 0))

            # Transparent mask
            mask_layer = overlay.copy()
            cv2.fillPoly(mask_layer, [pts], color)
            overlay = cv2.addWeighted(mask_layer, alpha, overlay, 1 - alpha, 0)

            # Polygon outline
            cv2.polylines(overlay, [pts], True, color, 2)

            # Label
            name = r.names.get(cls, "unknown")
            M = cv2.moments(pts)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            else:
                cx, cy = pts[0][0], pts[0][1]

            label = name
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            x1, y1 = max(0, cx - w // 2), max(0, cy - h - 10)
            x2, y2 = x1 + w + 6, y1 + h + 6

            # Translucent text background
            text_bg = overlay.copy()
            cv2.rectangle(text_bg, (x1, y1), (x2, y2), color, -1)
            overlay = cv2.addWeighted(text_bg, text_alpha, overlay, 1 - text_alpha, 0)

            # Adaptive text color
            brightness = np.mean(color)
            txt_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)

            cv2.putText(overlay, label, (x1 + 3, y1 + h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, txt_color, 2, cv2.LINE_AA)
    return overlay


def infer_frame(frame_bgr: np.ndarray, resize_to=None) -> np.ndarray:
    """Run YOLO segmentation + overlay."""
    if resize_to:
        frame_small = cv2.resize(frame_bgr, resize_to)
    else:
        frame_small = frame_bgr

    try:
        results = model.predict(frame_small, conf=0.6, verbose=False)
        return apply_segmentation_overlay(frame_small, results)
    except Exception as e:
        print(f"Inference error: {e}")
        return frame_small


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("WebSocket connected")
    try:
        while True:
            data = await ws.receive_bytes()
            npimg = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            loop = asyncio.get_running_loop()
            resize_to = (640, 480)
            annotated = await loop.run_in_executor(executor, infer_frame, frame, resize_to)

            if annotated is not None:
                success, jpeg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if success:
                    await ws.send_bytes(jpeg.tobytes())

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WS error: {e}")
        try:
            await ws.close()
        except:
            pass


if __name__ == "__main__":
    print(f"Starting server on port {PORT} ...")
    uvicorn.run("live_segmentation_ws:app", host="0.0.0.0", port=PORT, log_level="info")
