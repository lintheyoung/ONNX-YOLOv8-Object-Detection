import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from yolov8 import YOLOv8  # 确保你已经正确安装了YOLOv8及其依赖

class VideoMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'PyQt Video Perspective Transformation'
        self.left = 100
        self.top = 100
        self.width = 1280
        self.height = 960
        self.initUI()
        self.cap = cv2.VideoCapture(0)  # 使用第一个摄像头
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.frame_count = 0
        self.points = []
        self.transform_ready = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateFrame)
        self.timer.start(20)  # 设置定时器每20毫秒刷新一次
        self.scale_w = 1
        self.scale_h = 1

        # 初始化YOLOv8对象检测器
        model_path = "models/best.onnx"  # 根据实际模型路径调整
        self.yolov8_detector = YOLOv8(model_path, conf_thres=0.6, iou_thres=0.5)
        self.class_names = ['car_yellow', 'car_red', 'car_blue', 'car_green']  # 根据实际情况调整

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # 图像显示标签
        self.labelImage = QLabel(self)
        self.labelImage.move(20, 20)
        self.labelImage.resize(1200, 880)
        self.labelImage.setStyleSheet("border: 1px solid black;")
        self.labelImage.setMouseTracking(True)
        self.labelImage.mousePressEvent = self.getMousePosition

    def updateFrame(self):
        ret, frame = self.cap.read()
        if ret:
            # 旋转90度
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # 顺时针旋转90度

            self.frame_count += 1
            if self.frame_count == 10:
                self.original_img = frame.copy()
            if self.frame_count >= 10:
                self.displayImage(frame, transform=self.transform_ready)

    def displayImage(self, frame, transform=False):
        if transform and len(self.points) == 4:
            # 应用透视变换
            pts1 = np.float32(self.points)
            scale_num = 3
            pts2 = np.float32([[0, 0], [240 * scale_num, 0], [0, 300 * scale_num], [240 * scale_num, 300 * scale_num]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            frame = cv2.warpPerspective(frame, matrix, (240 * scale_num, 300 * scale_num))
            

            # 在这里执行对象检测
            boxes, scores, class_ids = self.yolov8_detector(frame)
            
            # 在frame上绘制检测结果
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                class_id = class_ids[i]
                score = scores[i]  # 获取当前框的得分
                class_name = self.class_names[class_id]
                # 确保坐标是整数类型
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"{class_name}: {score:.2f}"  # 显示类名和得分，保留两位小数
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # 计算中心点坐标
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                center_label = f"({cx:.0f}, {cy:.0f})"  # 格式化中心点坐标，保留整数部分
                cv2.putText(frame, center_label, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # 中心点坐标标记用红色表示


            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 在这里添加BGR到RGB的转换

        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytesPerLine = ch * w
        qImg = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)

        # 按比例缩放
        pixmap = pixmap.scaled(self.labelImage.width(), self.labelImage.height(), Qt.KeepAspectRatio)

        # 更新缩放比例
        self.scale_w = w / pixmap.width()
        self.scale_h = h / pixmap.height()

        self.labelImage.setPixmap(pixmap)


    def getMousePosition(self, event):
        if not self.transform_ready and self.frame_count >= 10 and len(self.points) < 4:
            x = int(event.pos().x() * self.scale_w)
            y = int(event.pos().y() * self.scale_h)
            self.points.append((x, y))
            if len(self.points) == 4:
                self.transform_ready = True
            print("Selected points:", self.points)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoMainWindow()
    window.show()
    sys.exit(app.exec_())
