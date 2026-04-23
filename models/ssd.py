# Ronald Zhang
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

#COCO class 1-based
CLASSES = {77: 'cell phone', 1: 'person', 55: 'orange'}

# SSD MobileNetV2 pretrained model loaded via TensorFlow Hub
# Architecture reference: https://github.com/tensorflow/models/tree/master/research/object_detection
model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.putText(frame, "Press Q to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = tf.expand_dims(tf.cast(rgb, tf.uint8), 0)

    results = model(input_tensor)
    boxes = results['detection_boxes'][0].numpy()
    scores = results['detection_scores'][0].numpy()
    classes = results['detection_classes'][0].numpy().astype(int)

    indices = tf.image.non_max_suppression(
        boxes, scores, max_output_size=50, iou_threshold=0.3, score_threshold=0.6
    ).numpy()
    boxes = boxes[indices]
    scores = scores[indices]
    classes = classes[indices]

    h, w = frame.shape[:2]

    for i in range(len(scores)):
        if classes[i] not in CLASSES:
            continue

        ymin, xmin, ymax, xmax = boxes[i]
        x1, y1, x2, y2 = int(xmin*w), int(ymin*h), int(xmax*w), int(ymax*h)

        label = f"{CLASSES[classes[i]]} {scores[i]:.2f}"
        cv2.rectangle(frame, (x1, y1,), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('SSD', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()