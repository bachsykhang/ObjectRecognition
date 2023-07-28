import cv2
import urllib.request
import pyttsx3
import threading

# URL của tệp XML haarcascade_car.xml trên GitHub
url = 'https://raw.githubusercontent.com/andrewssobral/vehicle_detection_haarcascades/master/cars.xml'

# Tên tệp XML lưu trữ tại máy tính của bạn
file_name = 'haarcascade_car.xml'

# Tải tệp XML từ GitHub và lưu nó vào máy tính
urllib.request.urlretrieve(url, file_name)

# Load Haar Cascade Classifier cho xe cộ
car_cascade = cv2.CascadeClassifier(file_name)

# Đọc video từ file hoặc stream video từ webcam
video_capture = cv2.VideoCapture('Xeco.mp4')  # Thay đổi đường dẫn nếu sử dụng video từ file

# Khởi tạo đối tượng pyttsx3 cho giọng nói
engine = pyttsx3.init()

# Vị trí nguy hiểm dưới vị trí trục x (đơn vị: pixel)
danger_zone_y = 300

# Hàm để đọc cảnh báo bằng giọng nói mà không làm dừng frame
def speak_warning():
    # Phát ra cảnh báo bằng giọng nói
    engine.say("wanning")
    engine.runAndWait()

while True:
    # Đọc từng frame từ video
    ret, frame = video_capture.read()
    if not ret:
        break

    # Giảm kích thước frame để tăng tốc độ xử lý
    resized_frame = cv2.resize(frame, (640, 360))  # Giảm kích thước frame xuống còn 640x360

    # Chuyển đổi ảnh sang ảnh xám để tăng tốc độ xử lý
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # Sử dụng Haar Cascade Classifier để nhận diện xe cộ trong frame
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))

    # Vẽ hình chữ nhật xung quanh các xe cộ nhận diện được
    for (x, y, w, h) in cars:
        cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Kiểm tra nếu vị trí y của khung ảnh nằm gần dưới vị trí trục x (nguy hiểm)
        if y + h >= danger_zone_y:
            # Sử dụng luồng riêng biệt để phát cảnh báo bằng giọng nói mà không làm dừng frame
            warning_thread = threading.Thread(target=speak_warning)
            warning_thread.start()

    # Hiển thị frame với các hình chữ nhật được vẽ xung quanh xe cộ
    cv2.imshow('Car Detection', resized_frame)

    # Thoát khỏi vòng lặp khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
video_capture.release()
cv2.destroyAllWindows()
