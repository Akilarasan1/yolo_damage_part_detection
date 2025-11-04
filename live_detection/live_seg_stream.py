import cv2
from flask import Flask, Response
from pyngrok import ngrok
from ultralytics import YOLO
import numpy as np

# ------------------ CONFIG ------------------
MODEL_PATH = "Yolo11_13oct_openvino_model/"  # your OpenVINO path
CAMERA_SOURCE = 0  # use 0 for webcam or IP stream URL
alpha = 0.45
text_alpha = 0.6

class_names = {
    0: 'crack', 1: 'dent', 2: 'glass shatter',
    3: 'lamp broken', 4: 'scratch', 5: 'tire flat'
}
class_colors = {
    0: (0, 0, 255), 1: (255, 0, 0), 2: (0, 255, 255),
    3: (255, 255, 0), 4: (0, 255, 0), 5: (255, 0, 255)
}
# ---------------------------------------------

print("🚀 Loading model...")
model = YOLO(MODEL_PATH, task="segment")
print("✅ Model loaded.")

app = Flask(__name__)

def generate_frames():
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print("❌ Cannot open camera.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(frame, verbose=False, device="cpu")
        overlay = frame.copy()

        for r in results:
            if r.masks is None:
                continue
            for mask, cls in zip(r.masks.xy, r.boxes.cls.int().tolist()):
                pts = np.array(mask, np.int32)
                color = class_colors.get(cls, (0, 255, 0))
                name = class_names.get(cls, "unknown")

                mask_layer = overlay.copy()
                cv2.fillPoly(mask_layer, [pts], color)
                overlay = cv2.addWeighted(mask_layer, alpha, overlay, 1 - alpha, 0)
                cv2.polylines(overlay, [pts], True, color, 2)

                M = cv2.moments(pts)
                if M["m00"] != 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                else:
                    cx, cy = pts[0][0], pts[0][1]

                label = name
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                x1, y1 = max(0, cx - w // 2), max(0, cy - h - 10)
                x2, y2 = x1 + w + 6, y1 + h + 6

                text_bg = overlay.copy()
                cv2.rectangle(text_bg, (x1, y1), (x2, y2), color, -1)
                overlay = cv2.addWeighted(text_bg, text_alpha, overlay, 1 - text_alpha, 0)
                brightness = np.mean(color)
                txt_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
                cv2.putText(overlay, label, (x1 + 3, y1 + h),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, txt_color, 2, cv2.LINE_AA)

        _, buffer = cv2.imencode('.jpg', overlay)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    public_url = ngrok.connect(5000)
    print(f"Public stream URL: {public_url}/")
    app.run(host='0.0.0.0', port=5000)
