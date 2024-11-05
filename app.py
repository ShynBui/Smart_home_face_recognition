import os

import cv2
import time
import serial
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from app import FaceDB, YOLOFaceDetector, get_docs_link, get_cam_link
import torch
import uvicorn
from collections import Counter
from threading import Thread, Event
from playsound import playsound
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Khởi tạo kết nối serial với Arduino (cập nhật "COM7" thành cổng của Arduino)
arduino = serial.Serial('COM7', 9600, timeout=1)

app = FastAPI()

# Khởi tạo VectorDBTracker và YOLOFaceDetector
vector_tracker = FaceDB()
face_detector = YOLOFaceDetector()
torch.device('cuda')
print("CUDA available:", torch.cuda.is_available())

# Biến trạng thái
detect_faces_enabled = False
start_detection_time = None
frame_count = 0
person_name_counts = Counter()
unknown_detected = False
door_unlocked = False
unlock_time = None
warning_active = False

# Event để kiểm soát âm thanh cảnh báo
warning_event = Event()

def play_warning_sound():
    global warning_active
    while True:
        warning_event.wait()  # Chờ cho đến khi cảnh báo được kích hoạt
        playsound(os.path.join(os.getcwd(), "alert_sound.mp3"))  # Đường dẫn đến tệp âm thanh cảnh báo
        warning_event.clear()  # Đợi đến lần kích hoạt tiếp theo
        warning_active = False


# Khởi chạy luồng âm thanh cảnh báo ngay khi ứng dụng chạy
Thread(target=play_warning_sound, name="WarningSoundThread", daemon=True).start()


def draw_text_pil(img, text, position, font_path="arial.ttf", font_size=20, color=(0, 255, 0)):
    # Convert the OpenCV image (BGR) to RGB
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Load a font supporting Vietnamese characters
    font = ImageFont.truetype(font_path, font_size)

    # Draw the text on the image
    draw.text(position, text, font=font, fill=color)

    # Convert the PIL image back to OpenCV (BGR format)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

@app.get("/toggle_detection")
async def toggle_detection():
    global detect_faces_enabled, start_detection_time, frame_count, person_name_counts, unknown_detected, door_unlocked, unlock_time, warning_active
    detect_faces_enabled = not detect_faces_enabled
    if detect_faces_enabled:
        start_detection_time = time.time()
        frame_count = 0
        person_name_counts.clear()
        unknown_detected = False
        door_unlocked = False
        unlock_time = None
        warning_active = False
    status = "ON" if detect_faces_enabled else "OFF"
    return {"Face Detection": status}


def generate_frames():
    global detect_faces_enabled, start_detection_time, frame_count, person_name_counts, unknown_detected, door_unlocked, unlock_time, warning_active
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        success, frame = cap.read()
        if not success or frame is None:
            continue

        if detect_faces_enabled and (time.time() - start_detection_time >= 7):
            detect_faces_enabled = False
            start_detection_time = None

        rectangle_color = (0, 255, 0) if detect_faces_enabled else (0, 0, 255)
        status_text = "Face Detection: ON" if detect_faces_enabled else "Face Detection: OFF"
        status_color = (0, 0, 255) if detect_faces_enabled else (255, 0, 0)

        if detect_faces_enabled:
            box_x_min, box_y_min = int(640 / 2 - 150), int(480 / 2 - 180)
            box_x_max, box_y_max = int(640 / 2 + 150), int(480 / 2 + 180)
            roi_frame = frame[box_y_min:box_y_max, box_x_min:box_x_max]

            faces = face_detector.detect_faces(roi_frame)

            for face in faces:
                x_min, y_min, x_max, y_max = face["bounding_box"]

                x_min += box_x_min
                y_min += box_y_min
                x_max += box_x_min
                y_max += box_y_min

                if x_min >= box_x_min and y_min >= box_y_min and x_max <= box_x_max and y_max <= box_y_max:
                    face_frame = frame[y_min:y_max, x_min:x_max]
                    person_name, flag = vector_tracker.track(face_frame)

                    if person_name == "unknown":
                        unknown_detected = True

                    person_name_counts[person_name] += 1
                    frame_count += 1

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (224, 144, 66), 2)
                    frame = draw_text_pil(frame, f"{person_name}", (x_min, y_min - 15), font_size=15, color=(0, 255, 0))

                    if frame_count >= 10:
                        most_common_person, count = person_name_counts.most_common(1)[0]
                        if count / frame_count >= 0.9 and most_common_person != 'unknown' and not door_unlocked:
                            rectangle_color = (0, 0, 255)
                            status_text = f"Mở khóa: {most_common_person}"
                            arduino.write(b'O')
                            door_unlocked = True
                            unlock_time = time.time()
                        elif count / frame_count < 0.9 or most_common_person == 'unknown':
                            warning_active = True
                            warning_event.set()  # Kích hoạt âm thanh cảnh báo

        if warning_active:
            rectangle_color = (0, 0, 255) if time.time() % 1 > 0.5 else (0, 255, 255)
            status_text = "Warning: Unauthorized Access"
            status_color = (0, 0, 255) if time.time() % 1 > 0.5 else (0, 255, 255)

        cv2.rectangle(frame, (int(640 / 2 - 150), int(480 / 2 - 180)), (int(640 / 2 + 150), int(480 / 2 + 180)),
                      rectangle_color, 2)
        frame = draw_text_pil(frame, status_text, (10, 30), font_size=20, color=status_color)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.on_event("shutdown")
async def shutdown_event():
    if 'cap' in globals():
        globals()['cap'].release()
    if arduino.is_open:
        arduino.close()


if __name__ == "__main__":
    print(f"Đường dẫn đến camera:{get_cam_link()}")
    print(f"Đường dẫn kích hoạt:{get_docs_link()}")
    uvicorn.run(app, host="0.0.0.0", port=8000)

