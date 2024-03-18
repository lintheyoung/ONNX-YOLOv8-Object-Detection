import cv2
import time
from yolov8 import YOLOv8

# 假设的类别名称列表，根据您的模型具体情况进行调整
class_names = ['car_yellow', 'car_red', 'car_blue', 'car_green']

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Initialize YOLOv8 object detector
model_path = "models/best.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

# Predefine window size while maintaining 1080P aspect ratio
window_name = "Detected Objects"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 960, 540)

fps_start_time = 0
fps = 0

while cap.isOpened():
    # Read frame from the video
    ret, frame = cap.read()
    if not ret:
        break
    
    # Start time of frame processing
    if fps_start_time == 0:
        fps_start_time = time.time()
    else:
        # Calculate FPS
        fps = 1 / (time.time() - fps_start_time)
        fps_start_time = time.time()

    # Detect objects in the frame
    boxes, scores, class_ids = yolov8_detector(frame)

    # Draw detections, center points, and class names on the frame
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
        class_id = class_ids[i]

        # Draw center point
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

        # Display class name
        class_name = class_names[class_id]
        cv2.putText(frame, class_name, (center_x + 10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display FPS on the frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow(window_name, frame)

    # Press key q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
