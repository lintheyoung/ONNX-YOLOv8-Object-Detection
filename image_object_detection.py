import cv2
from imread_from_url import imread_from_url

from yolov8 import YOLOv8

# Initialize yolov8 object detector
model_path = "models/best.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.2, iou_thres=0.3)

# Read image
img_url = "https://dedemaker-1255717351.cos.ap-nanjing.myqcloud.com/%E6%96%87%E4%BB%B6%E4%BC%A0%E8%BE%93/merged_merged_4942.jpg"
img = imread_from_url(img_url)

# Detect Objects
boxes, scores, class_ids = yolov8_detector(img)

# Draw detections
combined_img = yolov8_detector.draw_detections(img)
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
cv2.imshow("Detected Objects", combined_img)
cv2.imwrite("doc/img/detected_objects.jpg", combined_img)
cv2.waitKey(0)
