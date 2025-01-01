import cv2
import os
import time

# Tạo thư mục "push_up" nếu nó chưa tồn tại
output_dir = 'sitting'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Mở webcam
cap = cv2.VideoCapture(0)
cnt = 1
# Kiểm tra nếu webcam có thể được mở
if not cap.isOpened():
    print("Không thể truy cập webcam")
    exit()

# Đặt thông số cho video
frame_width = int(cap.get(3))  # Chiều rộng khung hình
frame_height = int(cap.get(4))  # Chiều cao khung hình
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec video cho định dạng MP4
out = cv2.VideoWriter(os.path.join(output_dir, f'{output_dir}_{cnt}.mp4'), fourcc, 20.0, (frame_width, frame_height))

# Ghi lại video trong 7 giây
start_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc khung hình từ webcam")
        break

    # Ghi lại frame vào video
    out.write(frame)

    # Hiển thị video
    cv2.imshow('Recording...', frame)

    # Dừng ghi nếu đã qua 7 giây
    if time.time() - start_time > 7:
        break

# Giải phóng các tài nguyên
cap.release()
out.release()
cv2.destroyAllWindows()
