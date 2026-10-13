
import cv2
import mediapipe as mp
import yt_dlp
import tempfile
import os

# MediaPipe 손 인식 설정
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# 유튜브 영상 링크 입력
youtube_url = input("유튜브 동영상 링크를 입력하세요: ")

try:
    # 임시 폴더에 유튜브 영상 다운로드
    with tempfile.TemporaryDirectory() as temp_dir:

        ydl_opts = {
            "format": "bestvideo[ext=mp4][height<=480]/bestvideo[height<=480]/bestvideo",
            "outtmpl": os.path.join(temp_dir, "video.%(ext)s"),
            "noplaylist": True,
            "js_runtimes": {"deno": {}}
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            video_path = ydl.prepare_filename(info)

        # 다운로드한 영상 열기
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError("영상 파일을 열 수 없습니다.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / fps) if fps > 0 else 33

        # 손 인식 실행
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:

            while cap.isOpened():
                success, image = cap.read()

                if not success:
                    break

                # RGB 변환
                rgb_image = cv2.cvtColor(
                    image, cv2.COLOR_BGR2RGB
                )

                # 손 인식
                results = hands.process(rgb_image)

                # 손 관절 표시
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS
                        )

                # 영상 출력
                cv2.imshow("YouTube Hand Detection", image)

                # ESC 누르면 종료
                if cv2.waitKey(delay) & 0xFF == 27:
                    break

        cap.release()

except Exception as e:
    print("오류 발생:", e)

finally:
    cv2.destroyAllWindows()
