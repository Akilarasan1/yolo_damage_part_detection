import cv2
import numpy as np
from ultralytics import YOLO

# Load model
# model = YOLO("Yolo11_13oct.pt")
model = YOLO("Yolo11_13oct_openvino_model/", task="segment")   # use your segmentation model (.pt)
# Class names and colors (BGR)
class_names = {
    0: 'crack',
    1: 'dent',
    2: 'glass shatter',
    3: 'lamp broken',
    4: 'scratch',
    5: 'tire flat'
}
class_colors = {
    0: (255,99,71),
    1: (0,191,255),
    2: (255,215,0),
    3: (255,215,0),
    4: (147,112,219),
    5: (255,215,0)
}

alpha = 0.45       # transparency for mask
text_alpha = 0.6   # transparency for text background

# Open webcam (0 for default cam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Cannot open camera.")
    exit()

print("✅ Camera opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not received. Exiting...")
        break

    # Run inference
    results = model.predict(frame, verbose=False, stream = True, device="cpu")
    # results = model.predict(frame, verbose=False, device="cuda")

    overlay = frame.copy()

    for r in results:
        if r.masks is None:
            continue
        for mask, cls in zip(r.masks.xy, r.boxes.cls.int().tolist()):
            pts = np.array(mask, np.int32)
            color = class_colors.get(cls, (0, 255, 0))
            name = class_names.get(cls, "unknown")

            # Draw filled transparent mask
            mask_layer = overlay.copy()
            cv2.fillPoly(mask_layer, [pts], color)
            overlay = cv2.addWeighted(mask_layer, alpha, overlay, 1 - alpha, 0)

            # Polygon outline
            cv2.polylines(overlay, [pts], True, color, 2)

            # Label placement
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

            # Text color (white/black based on brightness)
            brightness = np.mean(color)
            txt_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)

            # Draw label text
            cv2.putText(overlay, label, (x1 + 3, y1 + h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, txt_color, 2, cv2.LINE_AA)

    # Show real-time result
    cv2.imshow("YOLOv11 Real-Time Segmentation", overlay)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Stream ended.")
