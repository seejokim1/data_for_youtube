# =====================================================
# LANE DETECTION + LEFT/RIGHT + STEERING + SAVE VIDEO
# =====================================================

import cv2
import numpy as np
import os

# -----------------------------
# 1. 차선 검출 함수
# -----------------------------
def process_lane_detection(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

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

    lines = cv2.HoughLinesP(roi, 1, np.pi/180, 50,
                           minLineLength=50, maxLineGap=150)

    left_lines = []
    right_lines = []

    # -----------------------------
    # 2. 좌/우 차선 분리
    # -----------------------------
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            if x2 - x1 == 0:
                continue

            slope = (y2 - y1) / (x2 - x1)

            if abs(slope) < 0.5:
                continue

            if slope < 0:
                left_lines.append(line[0])
            else:
                right_lines.append(line[0])

    # -----------------------------
    # 3. 대표선 계산
    # -----------------------------
    def average_line(lines):
        if len(lines) == 0:
            return None

        x_coords = []
        y_coords = []

        for x1, y1, x2, y2 in lines:
            x_coords += [x1, x2]
            y_coords += [y1, y2]

        poly = np.polyfit(y_coords, x_coords, 1)  # x = my + b
        return poly

    left_fit = average_line(left_lines)
    right_fit = average_line(right_lines)

    line_img = np.zeros_like(frame)

    y1 = h
    y2 = int(h * 0.6)

    def make_points(poly):
        if poly is None:
            return None

        m, b = poly
        x1 = int(m * y1 + b)
        x2 = int(m * y2 + b)
        return (x1, y1, x2, y2)

    left_points = make_points(left_fit)
    right_points = make_points(right_fit)

    # 좌/우 차선 그리기
    if left_points is not None:
        cv2.line(line_img,
                 (left_points[0], left_points[1]),
                 (left_points[2], left_points[3]),
                 (255, 0, 0), 5)

    if right_points is not None:
        cv2.line(line_img,
                 (right_points[0], right_points[1]),
                 (right_points[2], right_points[3]),
                 (0, 255, 0), 5)

    # -----------------------------
    # 4. Steering Angle 계산
    # -----------------------------
    steering_angle = 0

    if left_points is not None and right_points is not None:

        mid_x1 = int((left_points[0] + right_points[0]) / 2)
        mid_x2 = int((left_points[2] + right_points[2]) / 2)

        # 중심선
        cv2.line(line_img, (mid_x1, y1), (mid_x2, y2), (0,255,255), 3)

        center_x = w // 2
        lane_center = mid_x1

        offset = lane_center - center_x

        steering_angle = np.degrees(np.arctan(offset / (h/2)))

        # 텍스트 표시
        cv2.putText(frame,
                    f"Steering: {int(steering_angle)} deg",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2)

    result = cv2.addWeighted(frame, 0.8, line_img, 1, 0)

    return result


# -----------------------------
# 5. 영상 입력
# -----------------------------
video_path = "andong_lane.mp4"  # ← 파일명 수정

if not os.path.exists(video_path):
    print("❌ 파일 없음:", video_path)
    exit()

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ 영상 열기 실패")
    exit()

# -----------------------------
# 6. 출력 영상 설정
# -----------------------------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter("lane_detection_left_right_seperate.mp4",
                      fourcc,
                      fps,
                      (width, height))

# -----------------------------
# 7. 프레임 처리
# -----------------------------
while True:
    ret, frame = cap.read()

    if not ret:
        print("영상 종료")
        break

    result = process_lane_detection(frame)

    # 화면 출력
    cv2.imshow("Lane Detection", result)

    # 🔥 영상 저장
    out.write(result)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# -----------------------------
# 8. 종료
# -----------------------------
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ 저장 완료: lane_detection_left_right_seperate.mp4")