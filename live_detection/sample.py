import cv2
import numpy as np
from ultralytics import YOLO

# Load model
model = YOLO("Yolo11_13oct.pt")

# Run inference
results = model.predict("sample.png")

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
    0: (0, 0, 255),
    1: (255, 0, 0),
    2: (0, 255, 255),
    3: (255, 255, 0),
    4: (0, 255, 0),
    5: (255, 0, 255)
}

alpha = 0.45  # transparency for mask
text_alpha = 0.6  # transparency for text background

for r in results:
    img = r.orig_img.copy()
    overlay = img.copy()

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

        # Text color (white or black based on mask brightness)
        brightness = np.mean(color)
        txt_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)

        # Draw label text
        cv2.putText(overlay, label, (x1 + 3, y1 + h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, txt_color, 2, cv2.LINE_AA)

    cv2.imshow("YOLO Segmentation Mask", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
