# =====================================================
# LANE DETECTION + VIDEO SAVE (FINAL VERSION)
# =====================================================

import cv2
import numpy as np
import os

# -----------------------------
# 1. Lane Detection 함수
# -----------------------------
def process_lane_detection(frame):

    # Gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 50, 150)

    # ROI 설정
    h, w = edges.shape
    mask = np.zeros_like(edges)

    polygon = np.array([[
        (int(w*0.1), h),
        (int(w*0.9), h),
        (int(w*0.6), int(h*0.6)),
        (int(w*0.4), int(h*0.6))
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    roi = cv2.bitwise_and(edges, mask)

    # Hough Transform
    lines = cv2.HoughLinesP(
        roi,
        1,
        np.pi/180,
        50,
        minLineLength=50,
        maxLineGap=150
    )

    # 라인 그리기
    line_img = np.zeros_like(frame)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 5)

    # Overlay
    result = cv2.addWeighted(frame, 0.8, line_img, 1, 0)

    return result


# -----------------------------
# 2. 입력 영상 경로 설정
# -----------------------------
video_path = "andong_lane.mp4"
# 👉 문제 있으면 영어 파일명으로 변경 권장: andong.mp4

if not os.path.exists(video_path):
    print("❌ 파일 없음:", video_path)
    exit()


# -----------------------------
# 3. 영상 열기
# -----------------------------
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ 영상 열기 실패")
    exit()


# -----------------------------
# 4. 출력 영상 설정
# -----------------------------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter("lane_detection.mp4", fourcc, fps, (width, height))


# -----------------------------
# 5. 프레임 처리 루프
# -----------------------------
while True:
    ret, frame = cap.read()

    if not ret:
        print("영상 종료")
        break

    # 차선 인식
    result = process_lane_detection(frame)

    # 화면 출력
    cv2.imshow("Lane Detection", result)

    # 영상 저장
    out.write(result)

    # ESC 키 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break


# -----------------------------
# 6. 종료 처리
# -----------------------------
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ 저장 완료: lane_detection.mp4")