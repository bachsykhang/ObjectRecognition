from flask import Flask, render_template, Response
import cv2
import pyttsx3
import threading
import numpy as np

app = Flask(__name__)

# Khởi tạo đối tượng pyttsx3 cho giọng nói
engine = pyttsx3.init()

# Hàm để đọc cảnh báo bằng giọng nói mà không làm dừng frame
def speak_warning():
    # Lấy đối tượng engine mới cho mỗi lần gọi hàm
    engine = pyttsx3.init()
    # Phát ra cảnh báo bằng giọng nói
    engine.say("Watch out for warnings")
    engine.setProperty('volume', 4)
    engine.runAndWait()

# Đoạn mã Python chạy trên video với YOLO
def process_video():
    # Load YOLO
    net = cv2.dnn.readNet('yolov3-tiny.cfg', 'yolov3-tiny.weights')
    classes = []
    with open('coco.names', 'r') as f:
        classes = f.read().strip().split('\n')

    # Đọc video từ file hoặc stream video từ webcam
    video_capture = cv2.VideoCapture('video/video2.mp4')  

    # Vị trí nguy hiểm dưới vị trí trục x (đơn vị: pixel)
    danger_zone_y = 280
    # Khoảng cách tối đa để phát hiện đối tượng xe
    max_distance = 150 

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Giảm kích thước frame để tăng tốc độ xử lý
        resized_frame = cv2.resize(frame, (640, 360))  # Giảm kích thước frame xuống còn 640x360

        # Chuẩn hóa ảnh, tạo blob và lấy thông tin khung hình
        blob = cv2.dnn.blobFromImage(resized_frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        # Xác định các tên output layers của mô hình
        out_layer_names = net.getUnconnectedOutLayersNames()

        # Sử dụng các tên output layers để lấy kết quả
        outs = net.forward(out_layer_names)

        # Xử lý kết quả từ YOLO để vẽ hình chữ nhật và phát cảnh báo
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and classes[class_id] == 'car':
                    center_x = int(detection[0] * resized_frame.shape[1])
                    center_y = int(detection[1] * resized_frame.shape[0])
                    w = int(detection[2] * resized_frame.shape[1])
                    h = int(detection[3] * resized_frame.shape[0])
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    # Chỉ vẽ đối tượng và phát cảnh báo nếu nó ở gần trung tâm khung hình
                    if abs(center_y + h/2 - resized_frame.shape[0]/2) <= max_distance:
                        cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        # Kiểm tra nếu vị trí y của khung ảnh nằm gần dưới vị trí trục x (nguy hiểm)
                        if center_y + h >= danger_zone_y:
                            # Sử dụng luồng riêng biệt để phát cảnh báo bằng giọng nói mà không làm dừng frame
                            warning_thread = threading.Thread(target=speak_warning)
                            warning_thread.start()
                            # Vẽ một khung đổ khi có cảnh báo
                            cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 0, 225), 2)

        # Gửi frame đã xử lý dưới dạng byte để hiển thị trên trang web
        ret, buffer = cv2.imencode('.jpg', resized_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Giải phóng tài nguyên
    video_capture.release()


@app.route('/')
def index():
    return render_template('index.html')  # Trả về file HTML để hiển thị trên trang web

# Route để hiển thị video
def gen(camera):
    return Response(process_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def video_feed():
    return gen(cv2.VideoCapture('video/video2.mp4'))

if __name__ == '__main__':
    app.run()
