import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from yolov8 import YOLOv8  # 确保你已经正确安装了YOLOv8及其依赖
import serial
import json
import time
from PyQt5.QtWidgets import QMessageBox

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QDialog, QVBoxLayout, QComboBox, QDialogButtonBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer

import serial.tools.list_ports  # 导入列出串口的库

# 打开串口
# ser = serial.Serial('COM8', 115200, timeout=0.1)  # 替换'COM端口号'为实际的端口号，比如'COM3'

# 初始化对话框，让用户选择摄像头和串口
class InitDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("选择摄像头和串口")
        self.layout = QVBoxLayout()

        # 创建摄像头下拉菜单
        self.cameraComboBox = QComboBox()
        self.layout.addWidget(self.cameraComboBox)
        self.populateCameraChoices()

        # 创建串口下拉菜单
        self.serialComboBox = QComboBox()
        self.layout.addWidget(self.serialComboBox)
        self.populateSerialChoices()

        # 创建对话框按钮
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.layout.addWidget(self.buttonBox)

        self.setLayout(self.layout)

    # 填充摄像头选择
    def populateCameraChoices(self):
        self.cameraComboBox.addItem("无")  # 添加“无”选项
        # 简单的方法是尝试打开摄像头，直到失败。这里假设最多尝试10个ID。
        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                self.cameraComboBox.addItem(f"摄像头 {i}")
                cap.release()
            else:
                break

    # 填充串口选择
    def populateSerialChoices(self):
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.serialComboBox.addItem(port.device)


number = 0

class VideoMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 初始化YOLOv8对象检测器之前
        msgBox = QMessageBox()

        # 在这里调用初始化对话框
        initDialog = InitDialog()

        self.title = '摄像头识别位置识别程序（VerCalico）'
        self.left = 100
        self.top = 100
        self.width = 1280
        self.height = 960
        self.initUI()


        if initDialog.exec_():
            cameraIndex = initDialog.cameraComboBox.currentIndex() - 1  # 调整index，因为新增了“无”
            serialPort = initDialog.serialComboBox.currentText()

            msgBox.setWindowTitle("模型加载中")
            msgBox.setText("正在加载对象检测模型，请稍候...")
            msgBox.setStandardButtons(QMessageBox.NoButton)  # 不显示任何标准按钮
            msgBox.open()  # 显示提示框

            # 根据选择初始化摄像头和串口
            self.initCamera(cameraIndex)
            self.initSerial(serialPort)
        else:
            sys.exit()  # 如果用户取消，则退出程序

        

        # self.cap = cv2.VideoCapture(0)  # 使用第一个摄像头
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
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

        msgBox.close()  # 关闭提示框

    # 根据用户选择初始化摄像头
    def initCamera(self, index):
        if index == -1:  # 如果选择了“无”
            self.camera_index = -1
            self.btnInfo.show()  # 显示按钮
            self.btnExit.hide()
            return
        
        self.camera_index = index
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self.btnInfo.show()  # 隐藏按钮，如果选择了有效的摄像头
        self.btnExit.hide()
        
        # self.cap = cv2.VideoCapture(index)

    # 发送数据函数
    def send_data(self, data):
        self.ser.write(data.encode())  # 将字符串编码发送
        # time.sleep(0.2)  # 每隔0.2秒发送一次

    # 根据用户选择初始化串口
    def initSerial(self, port):
        self.ser = serial.Serial(port, 115200, timeout=0.1)

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setFixedSize(960, 960)

        # 图像显示标签
        self.labelImage = QLabel(self)
        self.labelImage.move(20, 20)
        self.labelImage.resize(1200, 880)
        self.labelImage.setStyleSheet("border: 1px solid black;")
        self.labelImage.setMouseTracking(True)
        self.labelImage.mousePressEvent = self.getMousePosition

        # 添加按钮
        self.btnInfo = QPushButton("机器人启动", self)
        self.btnInfo.move(20, 920)  # 设置按钮位置
        self.btnInfo.clicked.connect(self.showInfo)  # 连接信号

        self.btnExit = QPushButton("退出程序", self)
        self.btnExit.move(200, 920)  # 设置按钮位置
        self.btnExit.clicked.connect(self.exitApp)  # 连接信号

        self.btnInfo.hide()  # 默认隐藏
        self.btnExit.hide()  # 默认隐藏

    def showInfo(self):
        data = json.dumps({"red_x": -1, "red_y": -1, "green_x": -1, "green_y": -1, "blue_x": -1, "blue_y": -1, "yellow_x": -1, "yellow_y": -1})  # 根据需要修改数据
        self.send_data(data + '\n')  # 加上换行符，方便ESP32端解析
        print(data)
        

        # 读取来自ESP32的数据
        if self.ser.in_waiting:
            line = self.ser.readline().decode('utf-8').strip()
            if line:
                print("Received:", line)

            line = self.ser.readline().decode('utf-8').strip()
            if line:
                print("Received:", line)

        time.sleep(0.2)

        data = json.dumps({"red_x": -1, "red_y": -1, "green_x": -1, "green_y": -1, "blue_x": -1, "blue_y": -1, "yellow_x": -1, "yellow_y": -1})  # 根据需要修改数据
        self.send_data(data + '\n')  # 加上换行符，方便ESP32端解析
        print(data)
        

        # 读取来自ESP32的数据
        if self.ser.in_waiting:
            line = self.ser.readline().decode('utf-8').strip()
            if line:
                print("Received:", line)

            line = self.ser.readline().decode('utf-8').strip()
            if line:
                print("Received:", line)

        time.sleep(0.2)

        data = json.dumps({"red_x": -1, "red_y": -1, "green_x": -1, "green_y": -1, "blue_x": -1, "blue_y": -1, "yellow_x": -1, "yellow_y": -1})  # 根据需要修改数据
        self.send_data(data + '\n')  # 加上换行符，方便ESP32端解析
        print(data)
        

        # 读取来自ESP32的数据
        if self.ser.in_waiting:
            line = self.ser.readline().decode('utf-8').strip()
            if line:
                print("Received:", line)

            line = self.ser.readline().decode('utf-8').strip()
            if line:
                print("Received:", line)
        
        time.sleep(0.2)

        data = json.dumps({"red_x": -1, "red_y": -1, "green_x": -1, "green_y": -1, "blue_x": -1, "blue_y": -1, "yellow_x": -1, "yellow_y": -1})  # 根据需要修改数据
        self.send_data(data + '\n')  # 加上换行符，方便ESP32端解析
        print(data)
        

        # 读取来自ESP32的数据
        if self.ser.in_waiting:
            line = self.ser.readline().decode('utf-8').strip()
            if line:
                print("Received:", line)

            line = self.ser.readline().decode('utf-8').strip()
            if line:
                print("Received:", line)

        QMessageBox.information(self, "机器人", "机器人启动")

    def exitApp(self):
        data = json.dumps({"red_x": -1, "red_y": -1, "green_x": -1, "green_y": -1, "blue_x": -1, "blue_y": -1, "yellow_x": -1, "yellow_y": -1})  # 根据需要修改数据
        self.send_data(data + '\n')  # 加上换行符，方便ESP32端解析
        print(data)
        QMessageBox.information(self, "信息2", "这是一个信息2显示按钮。")

    def updateFrame(self):
        if self.camera_index == -1:  # 如果选择了“无”
            return  # 不进行任何操作，直接返回
    
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
        global number
        if transform and len(self.points) == 4:
            # 应用透视变换
            pts1 = np.float32(self.points)
            scale_num = 3
            # 720 900
            pts2 = np.float32([[0, 0], [240 * scale_num, 0], [0, 300 * scale_num], [240 * scale_num, 300 * scale_num]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            frame = cv2.warpPerspective(frame, matrix, (240 * scale_num, 300 * scale_num))
            

            # 在这里执行对象检测
            boxes, scores, class_ids = self.yolov8_detector(frame)
            
            # 显示后更新全局的变量
            red_x = 0
            red_y = 0
            yellow_x = 0
            yellow_y = 0
            green_x = 0
            green_y = 0
            blue_x = 0
            blue_y = 0

            # 定义面积的最小阈值
            min_area_threshold = 1500  # 根据需要调整这个值

            # 在frame上绘制检测结果之前，先筛选每个类别得分最高的框，且框的面积需要大于min_area_threshold
            max_scores = {}
            max_boxes = {}
            for i, box in enumerate(boxes):
                class_id = class_ids[i]
                score = scores[i]
                class_name = self.class_names[class_id]
                
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)  # 计算当前框的面积
                
                if area > min_area_threshold and (class_name not in max_scores or score > max_scores[class_name]):
                    max_scores[class_name] = score
                    max_boxes[class_name] = box

            # 现在max_boxes包含了每个类别得分最高的框（且面积大于min_area_threshold），接下来只绘制这些框
            for class_name, box in max_boxes.items():
                score = max_scores[class_name]
                x1, y1, x2, y2 = map(int, box)  # 确保坐标是整数类型
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"{class_name}: {score:.2f}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # 计算中心点坐标
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                cx_show = (cx / (240 * scale_num)) * 240
                cy_show = (cy / (300 * scale_num)) * 300

                cx = cx - 25
                
                center_label = f"({cx_show:.0f}, {cy_show:.0f})"
                cv2.putText(frame, center_label, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # 更新对应颜色车辆的坐标
                if class_name == "car_red":
                    red_x, red_y = cx_show, cy_show
                elif class_name == "car_yellow":
                    yellow_x, yellow_y = cx_show, cy_show
                elif class_name == "car_blue":
                    blue_x, blue_y = cx_show, cy_show
                elif class_name == "car_green":
                    green_x, green_y = cx_show, cy_show

            # 接下来是数据发送逻辑，保持不变

                
            # 自定义JSON数据
            # 发送所有的位置
            number += 1
            data = json.dumps({"red_x": red_x, "red_y": red_y, "green_x": green_x, "green_y": green_y, "blue_x": blue_x, "blue_y": blue_y, "yellow_x": yellow_x, "yellow_y": yellow_y, "counter": number})  # 根据需要修改数据
            self.send_data(data + '\n')  # 加上换行符，方便ESP32端解析
            
            print(data)

            


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

        # 读取来自ESP32的数据
        if self.ser.in_waiting:
            line = self.ser.readline().decode('utf-8').strip()
            if line:
                print("Received:", line)


    def getMousePosition(self, event):
        if not self.transform_ready and self.frame_count >= 10 and len(self.points) < 4:
            x = int(event.pos().x() * self.scale_w)
            y = int(event.pos().y() * self.scale_h)
            self.points.append((x, y))
            # 在这里添加消息框
            msg = QMessageBox()
            msg.setWindowTitle("点选择")
            msg.setText(f"选择点{len(self.points)}")
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            
            if len(self.points) == 4:
                self.transform_ready = True
            print("Selected points:", self.points)


if __name__ == '__main__':
    print("import success, running")
    app = QApplication(sys.argv)
    window = VideoMainWindow()
    window.show()
    sys.exit(app.exec_())