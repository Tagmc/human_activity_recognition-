import cv2
import mediapipe as mp
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import threading
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ActionRecognitionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ActionRecognitionLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        x, _ = self.lstm3(x)
        x = self.dropout(x)
        x = x[:, -1, :]  
        x = self.fc(x)
        return x

num_of_timesteps = 7
model = ActionRecognitionLSTM(input_size=132, hidden_size=128, num_classes=8).to(device)  
model.load_state_dict(torch.load(f'model/model_trained_new.pth',map_location=torch.device('cpu')))
model.eval()

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

def make_landmark_timestep(results):
    lm_list = []
    landmarks = results.pose_landmarks.landmark
    
    base_x = landmarks[0].x
    base_y = landmarks[0].y
    base_z = landmarks[0].z
    
    center_x = np.mean([lm.x for lm in landmarks])
    center_y = np.mean([lm.y for lm in landmarks])
    center_z = np.mean([lm.z for lm in landmarks])

    distances = [np.sqrt((lm.x - center_x)**2 + (lm.y - center_y)**2 + (lm.z - center_z)**2) for lm in landmarks[1:]]
    scale_factors = [1.0 / dist for dist in distances]

    lm_list.append(0.0)
    lm_list.append(0.0)
    lm_list.append(0.0)
    lm_list.append(landmarks[0].visibility)

    for lm, scale_factor in zip(landmarks[1:], scale_factors):
        lm_list.append((lm.x - base_x) * scale_factor)
        lm_list.append((lm.y - base_y) * scale_factor)
        lm_list.append((lm.z - base_z) * scale_factor)
        lm_list.append(lm.visibility)
    return lm_list

def draw_landmark_on_image(results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    h, w, c = img.shape
    bbox = []
    if results.pose_landmarks:
        for id, lm in enumerate(results.pose_landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            bbox.append([cx, cy])
        x_min, y_min = np.min(bbox, axis=0)
        x_max, y_max = np.max(bbox, axis=0)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    return img

def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (20, 50)
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img

def draw_fps_on_image(fps, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (20, 100)
    fontScale = 1
    fontColor = (255, 0, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img, f'FPS: {fps:.2f}',
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img

label = "Unknown"

# Hàm dự đoán
def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = torch.tensor(lm_list, dtype=torch.float32).unsqueeze(0).to(device)  # Chuyển đổi thành tensor PyTorch
    with torch.no_grad():
        results = model(lm_list)
    predicted_label_index = torch.argmax(results, dim=1).item()
    classes = ['handclapping','handwaving', 'Sitting', 'Standing', 'Walking', 'Walking_While_Reading_Book', 'Walking_While_Using_Phone', 'push_up']
    confidence = torch.max(torch.softmax(results, dim=1)).item()
    if confidence > 0.95:
        label = classes[predicted_label_index]
    else:
        label = "unknown"

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

lm_list = []
video_writer = None
prev_time = 0
frame_count = 0  
previous_label = "unknown"
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frameRGB)
    if results.pose_landmarks:
        lm = make_landmark_timestep(results)
        lm_list.append(lm)
        if len(lm_list) == num_of_timesteps:
            detect_thread = threading.Thread(target=detect, args=(model, lm_list,))
            detect_thread.start()
            lm_list = []
        frame = draw_landmark_on_image(results, frame)
    frame = draw_class_on_image(label, frame)

    # if label != previous_label and label != "unknown":
    #     folder_path = f'output/{label}'
    #     os.makedirs(folder_path, exist_ok=True)
    #     timestamp = time.strftime("%Y%m%d_%H%M%S")
    #     cv2.imwrite(f"{folder_path}/{timestamp}.png", frame)
    #     previous_label = label
    if label != previous_label and label != "unknown":
        if video_writer:
            video_writer.release() 
            if frame_count < 100:  
                try:
                    os.remove(video_path)  
                except PermissionError as e:
                    print(f"Error removing video: {e}")
            video_writer = None
        folder_path = f'output_video/{label}'
        os.makedirs(folder_path, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(folder_path, f"{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
        frame_count = 0
        previous_label = label


    if video_writer:
        video_writer.write(frame)
        frame_count += 1

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    frame = draw_fps_on_image(fps, frame)
    cv2.imshow("image", frame)
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
