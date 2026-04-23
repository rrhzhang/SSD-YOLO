# Ronald Zhang
import cv2
from ultralytics import YOLO

# COCO class ID to objects
CLASSES = {67: 'cell phone', 0: 'person', 49: 'orange'}

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, verbose=False, iou=0.4, conf=0.3)

    cv2.putText(frame, "Press Q to quit", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        if class_id not in CLASSES:
            continue

        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"{CLASSES[class_id]} {confidence:.2f}"
        cv2.rectangle(frame, (x1, y1,), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('YOLO', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()