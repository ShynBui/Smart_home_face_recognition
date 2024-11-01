import os
import torch
import cv2
import numpy as np
from typing import List, Dict, Any
from ultralytics import YOLO
import time


class YOLOFaceDetector:
    def __init__(self, weight_path: str = os.path.join(os.getcwd(),'model', 'yolov8n-face.pt')):
        """
        Khởi tạo mô hình YOLO với trọng số sẵn có, chuyển lên GPU nếu có.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(weight_path)
        self.model = self.build_model(weight_path)
        self.model.to(self.device)
        # self.model.eval()  # Đặt mô hình ở chế độ đánh giá để dự đoán

    def build_model(self, weight_path: str) -> Any:
        """
        Xây dựng mô hình YOLO để phát hiện khuôn mặt từ trọng số đã có.
        """
        # Kiểm tra và tải trọng số nếu cần
        if not os.path.isfile(weight_path):
            self.download_weights(weight_path)

        # Khởi tạo mô hình YOLO từ trọng số đã có
        return YOLO(weight_path)

    def download_weights(self, weight_path: str):
        """
        Tải trọng số từ URL nếu chưa tồn tại.
        """
        print(f"Tải trọng số từ {WEIGHT_URL}...")
        gdown.download(WEIGHT_URL, weight_path, quiet=False)
        print("Tải trọng số hoàn tất.")

    def detect_faces(self, img: np.ndarray, confidence_threshold: float = 0.6) -> List[Dict]:
        """
        Phát hiện khuôn mặt từ ảnh đầu vào.
        """
        faces = []

        # Lưu kích thước ảnh gốc
        original_height, original_width = img.shape[:2]
        print(f'original size:{original_height}, {original_width}')

        # Chuyển ảnh về RGB nếu cần
        img_rgb = img
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Dự đoán khuôn mặt với mô hình YOLO
        start_time = time.time()
        results = self.model.predict(
            img_rgb,
            conf=confidence_threshold,
            device=self.device,
            verbose=False,
            show=False
        )[0]

        print("Yolo_time:", time.time() - start_time)
        # Lấy kích thước ảnh được sử dụng trong dự đoán
        resized_height, resized_width = img_rgb.shape[:2]
        print(f"Resize: {resized_height}, {resized_width}")

        # Tính tỷ lệ thay đổi kích thước (scale)
        scale_x = original_width / resized_width
        scale_y = original_height / resized_height

        for result in results:
            if result.boxes is None or result.keypoints is None:
                continue

            # Lấy bounding box và confidence
            x, y, w, h = result.boxes.xywh.tolist()[0]
            confidence = result.boxes.conf.tolist()[0]

            if confidence < confidence_threshold:
                continue

            # Tính lại tọa độ bounding box trên ảnh gốc
            x_min = int((x - w / 2) * scale_x)
            y_min = int((y - h / 2) * scale_y)
            x_max = int((x + w / 2) * scale_x)
            y_max = int((y + h / 2) * scale_y)

            # Lấy tọa độ mắt trái và mắt phải, áp dụng tỷ lệ
            right_eye = tuple(map(lambda v: int(v * scale_x), result.keypoints.xy[0][0].tolist()))
            left_eye = tuple(map(lambda v: int(v * scale_y), result.keypoints.xy[0][1].tolist()))

            faces.append({
                "bounding_box": (x_min, y_min, x_max, y_max),
                "left_eye": left_eye,
                "right_eye": right_eye,
                "confidence": confidence
            })

        return faces


    def display_faces(self, img: np.ndarray, faces: List[Dict]) -> None:
        """
        Hiển thị ảnh với các bounding box và mắt của các khuôn mặt được phát hiện.

        Args:
            img (np.ndarray): Ảnh đầu vào.
            faces (List[Dict]): Danh sách các khuôn mặt được phát hiện.
        """
        for face in faces:
            x_min, y_min, x_max, y_max = face["bounding_box"]
            confidence = face["confidence"]

            # Vẽ bounding box lên ảnh
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(img, f"{confidence:.2f}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Hiển thị ảnh với các khuôn mặt được phát hiện
        cv2.imshow("Detected Faces", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

