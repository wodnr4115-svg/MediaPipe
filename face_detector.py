
import cv2
import mediapipe as mp

# MediaPipe 얼굴 인식 설정
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

face_detection = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

# 교수님이 제공한 얼굴 영상 사용
cap = cv2.VideoCapture("face.mp4")

while cap.isOpened():
    success, image = cap.read()

    if not success:
        break

    # 좌우 반전 및 RGB 변환
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 얼굴 인식
    results = face_detection.process(rgb_image)

    # 얼굴 위치 표시
    if results.detections:
        for detection in results.detections:
            mp_draw.draw_detection(image, detection)

    cv2.imshow("Face Detection", image)

    # ESC 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
face_detection.close()
cv2.destroyAllWindows()
