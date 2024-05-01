import serial
import json
import time

# 打开串口
ser = serial.Serial('COM57', 115200, timeout=0.1)  # 替换'COM端口号'为实际的端口号，比如'COM3'

# 发送数据函数
def send_data(data):
    ser.write(data.encode())  # 将字符串编码发送
    time.sleep(0.2)  # 每隔0.2秒发送一次

# 获取位置之后就发送ESP32上

# 主循环
try:
    while True:
        # 自定义JSON数据
        data = json.dumps({"red_x": 1, "red_y": 2})  # 根据需要修改数据
        send_data(data + '\n')  # 加上换行符，方便ESP32端解析
        
        # 读取来自ESP32的数据
        if ser.in_waiting:
            line = ser.readline().decode('utf-8').strip()
            if line:
                print("Received:", line)
except KeyboardInterrupt:
    ser.close()  # 退出时关闭串口
