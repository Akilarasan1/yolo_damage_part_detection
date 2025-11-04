import cv2
from ultralytics import YOLO

# Load as segmentation model explicitly
model = YOLO("Yolo11_13oct_openvino_model/")
model.overrides['task'] = 'segment'  # ensures proper segmentation mode

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera could not be opened.")
    exit()

# Optional: skip frames for speed
frame_skip = 2
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % frame_skip != 0:
        continue

    # Resize for smoother performance
    frame_resized = cv2.resize(frame, (640, 460))
    results = model.predict(frame_resized, conf=0.5, verbose=False)


    # Display segmentation mask overlay
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Damage Segmentation", annotated_frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
