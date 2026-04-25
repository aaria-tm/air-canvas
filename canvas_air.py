import cv2
import numpy as np
import mediapipe as mp

cap = cv2.VideoCapture(0)

canvas = None
prev_x, prev_y = 0, 0

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Colors
colors = {
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "red": (0, 0, 255)
}

current_color = "blue"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    h, w, _ = frame.shape

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm = hand_landmarks.landmark

            # Index finger tip
            x = int(lm[8].x * w)
            y = int(lm[8].y * h)

            # Finger detection
            index_up = lm[8].y < lm[6].y
            middle_up = lm[12].y < lm[10].y

            # ✏️ DRAW → only index finger
            if index_up and not middle_up:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                cv2.line(canvas, (prev_x, prev_y), (x, y), colors[current_color], 5)
                prev_x, prev_y = x, y

            # 🧽 ERASE → index + middle finger
            elif index_up and middle_up:
                cv2.circle(canvas, (x, y), 30, (0, 0, 0), -1)
                prev_x, prev_y = 0, 0

            else:
                prev_x, prev_y = 0, 0

            # Draw hand skeleton
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    else:
        prev_x, prev_y = 0, 0

    # Merge drawing with frame
    frame = cv2.add(frame, canvas)

    # Watermark
    cv2.putText(frame, "Made by Aarya", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Current color display
    cv2.putText(frame, f"Color: {current_color}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[current_color], 2)

    cv2.imshow("Canvas Air", frame)

    key = cv2.waitKey(1) & 0xFF

    # Controls
    if key == 27:  # ESC
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)
    elif key == ord('1'):
        current_color = "blue"
    elif key == ord('2'):
        current_color = "green"
    elif key == ord('3'):
        current_color = "red"

cap.release()
cv2.destroyAllWindows()