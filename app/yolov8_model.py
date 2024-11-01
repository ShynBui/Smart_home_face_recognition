import os
import torch
import cv2
import numpy as np
from typing import List, Dict, Any
from ultralytics import YOLO


class YOLOFaceDetector:
    def __init__(self, weight_path: str = "models/yolov8_weights.pth"):
        """
        Khởi tạo mô hình YOLO với trọng số sẵn có, chuyển lên GPU nếu có.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.build_model(weight_path)
        self.model.to(self.device)
        self.model.eval()  # Đặt mô hình ở chế độ đánh giá để dự đoán

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

    def detect_faces(self, img: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Phát hiện khuôn mặt từ ảnh đầu vào.

        Args:
            img (np.ndarray): Ảnh đầu vào dưới dạng numpy array.
            confidence_threshold (float): Ngưỡng độ tin cậy để lọc kết quả phát hiện.

        Returns:
            List[Dict]: Danh sách các khuôn mặt được phát hiện với thông tin vùng khuôn mặt.
        """
        faces = []

        # Chuyển ảnh về RGB nếu cần
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Dự đoán khuôn mặt với mô hình YOLO
        results = self.model.predict(
            img_rgb,
            conf=confidence_threshold,
            device=self.device,
            verbose=False
        )[0]

        # Duyệt qua từng khuôn mặt được phát hiện
        for result in results.boxes:
            if not result or not result.xyxy:
                continue

            # Lấy tọa độ bounding box và độ tin cậy
            x_min, y_min, x_max, y_max = map(int, result.xyxy[0].tolist())
            confidence = result.conf[0].item()

            # Bỏ qua phát hiện có confidence thấp
            if confidence < confidence_threshold:
                continue

            # Thêm khuôn mặt vào danh sách kết quả
            faces.append({
                "bounding_box": (x_min, y_min, x_max, y_max),
                "confidence": confidence
            })

        return faces
