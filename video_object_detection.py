import cv2
import time  # 导入time模块以测量时间

from yolov8 import YOLOv8

cap = cv2.VideoCapture("test_video.mp4")
start_time = 5
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))

model_path = "models/best.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

prev_time = 0  # 初始化前一次记录时间的变量
while cap.isOpened():
    if cv2.waitKey(1) == ord('q'):
        break

    try:
        ret, frame = cap.read()
        if not ret:
            break
    except Exception as e:
        print(e)
        continue

    # 在检测之前记录时间
    start_time = time.time()

    boxes, scores, class_ids = yolov8_detector(frame)
    combined_img = yolov8_detector.draw_detections(frame)
    
    # 在完成检测后记录时间，并计算帧率
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # 将帧率信息添加到视频帧上
    cv2.putText(combined_img, f"FPS: {fps:.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Detected Objects", combined_img)

# 清理工作
cap.release()
cv2.destroyAllWindows()
