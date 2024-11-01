import cv2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import time
from app import FaceDB, YOLOFaceDetector
import torch
import uvicorn

app = FastAPI()

# Khởi tạo VectorDBTracker và YOLOFaceDetector
vector_tracker = FaceDB()
face_detector = YOLOFaceDetector()
torch.device('cuda')
print("CUDA available:", torch.cuda.is_available())


def generate_frames():
    # Mở camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        success, frame = cap.read()
        if not success or frame is None:
            continue

        # Phát hiện khuôn mặt trong khung hình
        faces = face_detector.detect_faces(frame)

        # Nếu có khuôn mặt, tiến hành nhận diện từng khuôn mặt
        for face in faces:
            # print(face)

            # Lấy vị trí bounding box của khuôn mặt
            x_min, y_min, x_max, y_max = face["bounding_box"]
            print(face["bounding_box"])

            # print("Kích thước ảnh:", frame.shape)

            # Cắt khuôn mặt từ khung hình
            face_frame = frame[y_min:y_max, x_min:x_max]

            # cv2.imshow('Color image', face_frame)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # face_frame_rgb = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)

            # Kiểm tra danh tính trong cơ sở dữ liệu vector
            person_name, flag = vector_tracker.track(face_frame)

            if flag == None:
                # Nếu không tìm thấy trong DB, thêm khuôn mặt vào cơ sở dữ liệu với tên là "unknown"
                # person_name, stored_age = vector_tracker.add_to_db(face_frame_rgb, None)
                pass

            # Vẽ bounding box và thông tin nhận diện lên frame
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (224, 144, 66), 2)
            cv2.putText(frame, f"{person_name}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Chuyển đổi frame thành bytes để stream
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Đảm bảo giải phóng camera
    cap.release()


@app.get("/video_feed")
async def video_feed():
    # API cho video stream
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.on_event("shutdown")
async def shutdown_event():
    # Giải phóng tài nguyên khi tắt ứng dụng
    if 'cap' in globals():
        globals()['cap'].release()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)