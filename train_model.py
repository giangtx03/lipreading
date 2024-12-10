import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, TimeDistributed, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import json

# ========================================
# 1. Hàm xử lý dữ liệu
# ========================================

# Trích xuất frame từ video
def extract_frames(video_path, output_folder, fps=25):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_folder = os.path.join(output_folder, video_name)
    os.makedirs(video_folder, exist_ok=True)  # Tạo thư mục cho mỗi video
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % (frame_rate // fps) == 0:  # Lấy frame theo FPS mong muốn
            frame_name = f"frame_{count:04d}.jpg"
            frame_path = os.path.join(video_folder, frame_name)
            cv2.imwrite(frame_path, frame)
        count += 1
    cap.release()

# Trích xuất frame từ tất cả video trong thư mục
def extract_frames_from_videos(video_paths, frames_folder, fps=25):
    os.makedirs(frames_folder, exist_ok=True)
    for video_path in video_paths:
        extract_frames(video_path, frames_folder, fps)

# Cắt vùng miệng từ frame
def crop_mouth(frames_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=0.5) as face_mesh:
        for video_folder in os.listdir(frames_folder):
            video_folder_path = os.path.join(frames_folder, video_folder)
            if os.path.isdir(video_folder_path):
                for frame_file in sorted(os.listdir(video_folder_path)):
                    frame_path = os.path.join(video_folder_path, frame_file)
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
                    output_path = os.path.join(output_folder, video_folder, frame_file)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Tạo thư mục cho từng video
                    cv2.imwrite(output_path, mouth_resized)

# Chuyển frame thành ảnh xám
def preprocess_frames(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for video_folder in os.listdir(input_folder):
        video_folder_path = os.path.join(input_folder, video_folder)
        if os.path.isdir(video_folder_path):
            for frame_file in sorted(os.listdir(video_folder_path)):
                frame_path = os.path.join(video_folder_path, frame_file)
                frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
                output_path = os.path.join(output_folder, video_folder, frame_file)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Tạo thư mục cho từng video
                cv2.imwrite(output_path, frame)


# Tạo tensor dữ liệu từ folder frame
def create_video_tensor(folder, target_length=75):
    video_tensor = []
    for frame_file in sorted(os.listdir(folder))[:target_length]:  # Lấy tối đa target_length frame
        frame = cv2.imread(os.path.join(folder, frame_file), cv2.IMREAD_GRAYSCALE)
        video_tensor.append(frame)
    while len(video_tensor) < target_length:  # Padding nếu thiếu frame
        video_tensor.append(np.zeros((48, 48), dtype=np.uint8))
    return np.array(video_tensor)

def load_labels_and_map(labels_folder, video_paths):
    """
    Tải nhãn từ thư mục và tạo ánh xạ {int: string}.
    """
    labels = []
    unique_labels = set()  # Bộ lưu trữ các nhãn duy nhất
    for video_path in video_paths:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        label_file = os.path.join(labels_folder, f"{video_name}.txt")  # Đường dẫn file nhãn
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                label = f.read().strip()  # Đọc và loại bỏ khoảng trắng
                labels.append(label)
                unique_labels.add(label)  # Thêm nhãn vào bộ tập hợp
        except FileNotFoundError:
            print(f"Lỗi: Không tìm thấy tệp nhãn cho {video_name}")
            labels.append(None)
        except UnicodeDecodeError:
            print(f"Lỗi: Không thể đọc tệp {label_file} do vấn đề mã hóa")
            labels.append(None)

    # Tạo ánh xạ {int: string} từ bộ nhãn duy nhất
    label_map = {i: label for i, label in enumerate(sorted(unique_labels))}
    
    # Chuyển danh sách nhãn thành danh sách số nguyên dựa trên ánh xạ
    label_ids = [list(label_map.keys())[list(label_map.values()).index(label)] for label in labels if label]

    return label_ids, label_map

# ========================================
# 2. Xây dựng mô hình CNN + LSTM
# ========================================

def build_model(input_shape=(75, 48, 48, 1), num_classes=50):
    model = Sequential([
        TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=input_shape),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Flatten()),
        LSTM(128, return_sequences=False),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dense(256, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ========================================
# 3. Huấn luyện mô hình
# ========================================

def train_model(video_tensors, labels, num_classes=50, test_size=0.2, epochs=20, batch_size=16):
    labels = to_categorical(labels, num_classes=num_classes)
    X_train, X_val, y_train, y_val = train_test_split(video_tensors, labels, test_size=test_size, random_state=42)
    model = build_model(input_shape=(75, 48, 48, 1), num_classes=num_classes)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)
    return model

# ========================================
# 4. Chạy chương trình
# ========================================

if __name__ == "__main__":
    # Đường dẫn video và các folder lưu trữ
    video_folder = "videos"  # Thư mục chứa video
    labels_folder = "labels"  # Thư mục chứa nhãn
    frames_folder = "frames"
    mouth_folder = "mouth_frames"
    gray_folder = "gray_frames"

    # Lấy tất cả video từ thư mục videos
    video_paths = [os.path.join(video_folder, video_file) for video_file in os.listdir(video_folder)]

    # Bước 1: Trích xuất frame từ video
    extract_frames_from_videos(video_paths, frames_folder)

    # Bước 2: Cắt vùng miệng từ frame
    crop_mouth(frames_folder, mouth_folder)

    # Bước 3: Chuyển frame thành ảnh xám
    preprocess_frames(mouth_folder, gray_folder)

    # Bước 4: Tạo video tensor từ các frame đã xử lý
    video_tensors = []
    for video_folder in os.listdir(gray_folder):
        video_folder_path = os.path.join(gray_folder, video_folder)
        if os.path.isdir(video_folder_path):
            video_tensor = create_video_tensor(video_folder_path)
            video_tensors.append(video_tensor)

    # Bước 5: Tải nhãn và ánh xạ
    labels, label_map = load_labels_and_map(labels_folder, video_paths)

    # Lưu label_map vào tệp JSON
    with open('label_map.json', 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=4)
    
    # Bước 6: Huấn luyện mô hình
    video_tensors = np.array(video_tensors)
    model = train_model(video_tensors, labels, num_classes=len(label_map), epochs=20, batch_size=16)
    model.save('lipreading_model.h5')
    print("Huấn luyện hoàn thành!")
