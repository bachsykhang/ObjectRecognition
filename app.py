from flask import Flask, render_template, Response, request
import cv2
import pyttsx3
import threading
import numpy as np

app = Flask(__name__)


# Danh sách các vật thể để hệ thống nhận diện
with open('phuongTien.txt', 'r') as file:
    # Đọc nội dung của file và tách thành danh sách các phần tử
    transportName = file.read().strip().split('\n')
# Hàm để chụp và lưu ảnh
def capture_and_save_frame(frame):
    # Chụp ảnh từ khung hình hiện tại
    captured_frame = frame.copy()
    # Lưu ảnh vào file
    cv2.imwrite("Image.jpg", captured_frame)
    print("Đã chụp và lưu ảnh thành công.")
# Hàm để đọc cảnh báo bằng giọng nói mà không làm dừng frame
def speak_warning():
    # Lấy đối tượng engine mới cho mỗi lần gọi hàm
    engine = pyttsx3.init()
    # Phát ra cảnh báo bằng giọng nói
    engine.say("Watch out for warnings")
    engine.setProperty('volume', 4)
    engine.runAndWait()
# Đoạn mã Python chạy trên video với YOLO
def process_video(video_path):
    # Load YOLO
    net = cv2.dnn.readNet('yolov4-tiny.cfg', 'yolov4-tiny.weights')
    classes = []
    with open('coco.names', 'r') as f:
        classes = f.read().strip().split('\n')

    # Đọc video từ file hoặc stream video từ webcam
    video_capture = cv2.VideoCapture(video_path)  

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
        detected_objects = []
        # Xử lý kết quả từ YOLO để vẽ hình chữ nhật và phát cảnh báo
        detections = np.concatenate(outs)
        for detection in detections:
            class_id = np.argmax(detection[5:])
            confidence = detection[5:][class_id]
            if confidence > 0.5 and classes[class_id] in transportName:
                coco_class_index = transportName.index(classes[class_id])
                classes[class_id] = transportName[coco_class_index]
                frame_width = resized_frame.shape[1]
                frame_height = resized_frame.shape[0]
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                w = int(detection[2] * frame_width)
                h = int(detection[3] * frame_height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                 # Kiểm tra xem đối tượng đã được xác định trước đó hay chưa
                is_detected = False
                for obj in detected_objects:
                    if abs(center_x - obj[0]) <= w / 2 and abs(center_y - obj[1]) <= h / 2:
                        is_detected = True
                        break

                if not is_detected:
                    # Chỉ vẽ đối tượng và phát cảnh báo nếu nó ở gần trung tâm khung hình
                    if abs(center_y + h/2 - resized_frame.shape[0]/2) <= max_distance:
                        img = cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(resized_frame, classes[class_id], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        # Kiểm tra nếu vị trí y của khung ảnh nằm gần dưới vị trí trục x (nguy hiểm)
                        if (center_y + h) >= danger_zone_y:
                            # Sử dụng luồng riêng biệt để phát cảnh báo bằng giọng nói mà không làm dừng frame
                            warning_thread = threading.Thread(target=speak_warning)
                            warning_thread.start()
                            # Vẽ một khung đổ khi có cảnh báo
                            cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 0, 225), 2)
                            # Chụp và lưu ảnh
                            capture_and_save_frame(resized_frame)
                    # Lưu trữ thông tin về đối tượng đã được xác định
                    detected_objects.append((center_x, center_y))

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
def gen(video_path):
    return Response(process_video(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed', methods=['POST'])
def video_feed():
    if request.method == 'POST':
        video = request.files['video']
        video.save('video/video.mp4')
        return gen('video/video.mp4')
    return "No video file provided."


if __name__ == '__main__':
    app.run()