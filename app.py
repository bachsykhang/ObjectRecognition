from flask import Flask, render_template, Response
import cv2
import pyttsx3
import threading

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

# Đoạn mã Python chạy trên video
def process_video():
    # Load Haar Cascade Classifier cho xe cộ
    car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

    # Đọc video từ file hoặc stream video từ webcam
    video_capture = cv2.VideoCapture('video/video2.mp4')  

    # Vị trí nguy hiểm dưới vị trí trục x (đơn vị: pixel)
    danger_zone_y = 280
    # Khoảng cách tối đa để phát hiện đối tượng xe
    max_distance = 150 
    # Khai báo biến để lưu thông tin xe cộ
    vehicle_info = {}

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Giảm kích thước frame để tăng tốc độ xử lý
        resized_frame = cv2.resize(frame, (640, 360))  # Giảm kích thước frame xuống còn 640x360

        # Chuyển đổi ảnh sang ảnh xám để tăng tốc độ xử lý
        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

        # Sử dụng Haar Cascade Classifier để nhận diện xe cộ trong frame
        cars = car_cascade.detectMultiScale(gray, scaleFactor=1.09, minNeighbors=5, minSize=(35, 35))

        # Vẽ hình chữ nhật xung quanh các xe cộ nhận diện được
        for (x, y, w, h) in cars:
            # Tính khoảng cách từ tọa độ của đối tượng đến trung tâm khung hình
            distance = abs(x + w/2 - resized_frame.shape[1]/2)
            # Chỉ vẽ đối tượng và phát cảnh báo nếu nó ở gần trung tâm khung hình
            if distance <= max_distance:
                cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Kiểm tra nếu vị trí y của khung ảnh nằm gần dưới vị trí trục x (nguy hiểm)
                if y + h >= danger_zone_y:
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
    cv2.destroyAllWindows()

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
