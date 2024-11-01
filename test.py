import cv2
from app import YOLOFaceDetector


yolo_model = YOLOFaceDetector()

img = cv2.imread('D:\Smart_home_face_recognition\multi_people_img.jpg')

result = yolo_model.detect_faces(img=img)

yolo_model.display_faces(img=img, faces=result)