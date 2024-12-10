import json
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# ========================================
# 1. Hàm xử lý dữ liệu
# ========================================

def extract_frames(video_path, output_folder, fps=25):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % (frame_rate // fps) == 0:  # Lấy frame theo FPS mong muốn
            frame_name = f"{output_folder}/frame_{count:04d}.jpg"
            cv2.imwrite(frame_name, frame)
        count += 1
    cap.release()


def crop_mouth(frames_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=0.5) as face_mesh:
        for frame_file in sorted(os.listdir(frames_folder)):
            frame_path = os.path.join(frames_folder, frame_file)
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)
            if not results.multi_face_landmarks:
                continue
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame.shape

            mouth_indices = [0, 37, 267, 269, 270, 409, 78, 95, 88, 185, 40, 39, 37, 0, 267, 61, 146, 
                             91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317]
            mouth_coords = [(int(face_landmarks.landmark[i].x * w), 
                             int(face_landmarks.landmark[i].y * h)) for i in mouth_indices]
            x_min = max(0, min([p[0] for p in mouth_coords]))
            y_min = max(0, min([p[1] for p in mouth_coords]))
            x_max = min(w, max([p[0] for p in mouth_coords]))
            y_max = min(h, max([p[1] for p in mouth_coords]))
            mouth = frame[y_min:y_max, x_min:x_max]
            if mouth.size == 0:
                continue
            mouth_resized = cv2.resize(mouth, (48, 48))
            output_path = os.path.join(output_folder, frame_file)
            cv2.imwrite(output_path, mouth_resized)

def preprocess_frames(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for frame_file in sorted(os.listdir(input_folder)):
        frame_path = os.path.join(input_folder, frame_file)
        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(f"{output_folder}/{frame_file}", frame)

def create_video_tensor(folder, target_length=75):
    frames = sorted(os.listdir(folder))
    video_tensor = []
    for frame_file in frames[:target_length]:
        frame = cv2.imread(os.path.join(folder, frame_file), cv2.IMREAD_GRAYSCALE)
        video_tensor.append(frame)
    while len(video_tensor) < target_length:
        video_tensor.append(np.zeros((48, 48), dtype=np.uint8))
    return np.array(video_tensor)

def lipreading_to_text_multiple(video_path, model, label_map, target_length=75):
    frames_folder = "predict_frames"  # Folder cho dữ liệu dự đoán
    mouth_folder = "predict_mouth_frames"
    gray_folder = "predict_gray_frames"

    # Bước 1: Trích xuất frame từ video
    extract_frames(video_path, frames_folder)

    # Bước 2: Cắt vùng miệng từ frame
    crop_mouth(frames_folder, mouth_folder)

    # Bước 3: Chuyển frame thành ảnh xám
    preprocess_frames(mouth_folder, gray_folder)

    # Bước 4: Xử lý nhiều đoạn trong video (giả sử video có 2 đoạn: "anh" và "em")
    video_tensor = create_video_tensor(gray_folder)

    # Bước 5: Dự đoán văn bản từ video
    video_tensor = video_tensor[np.newaxis, ..., np.newaxis]  # Thêm chiều batch và kênh
    predictions = model.predict(video_tensor)

    # Ánh xạ dự đoán thành văn bản
    predicted_labels = np.argmax(predictions, axis=-1)
    
    # Dự đoán cho mỗi đoạn video
    texts = [label_map[str(label)] for label in predicted_labels]

    return " ".join(texts)

if __name__ == "__main__":
    # Đường dẫn video và mô hình đã huấn luyện
    video_path = "test.mp4"
    model = load_model('lipreading_model.h5')
    
    # Tải `label_map` từ tệp JSON
    with open('label_map.json', 'r', encoding='utf-8') as f:
        label_map = json.load(f)
        
    # Dự đoán văn bản
    text = lipreading_to_text_multiple(video_path, model, label_map)
    print("Văn bản dự đoán từ video: ", text)
