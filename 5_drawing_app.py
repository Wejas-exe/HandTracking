import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Initialize a blank canvas
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Open webcam
cap = cv2.VideoCapture(0)

# Previous coordinates
prev_x, prev_y = None, None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Extract hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get index finger tip (landmark 8)
            x, y = int(hand_landmarks.landmark[8].x * frame.shape[1]), int(hand_landmarks.landmark[8].y * frame.shape[0])
            
            # Get palm center (landmark 9) for more stable eraser position
            palm_x, palm_y = int(hand_landmarks.landmark[9].x * frame.shape[1]), int(hand_landmarks.landmark[9].y * frame.shape[0])

            # Check if index finger is raised (above middle finger)
            index_finger_up = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y

            middle_finger_up = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y

            # Improved Open Palm Detection (All fingers extended)
            fingers_extended = [
                hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y
                for tip in [8, 12, 16, 20]  # Tips of index, middle, ring, pinky fingers
            ]
            opened_palm = all(fingers_extended)

            # Drawing Mode
            if index_finger_up and not opened_palm:
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 255, 255), 5)
                prev_x, prev_y = x, y
            else:
                prev_x, prev_y = None, None

            # Eraser Mode (Open Palm)
            if opened_palm:
                cv2.circle(canvas, (palm_x, palm_y), 30, (0, 0, 0), -1)
            
            if middle_finger_up and not opened_palm :
                canvas[:] = 0

    # Merge drawing on the frame
    frame = cv2.addWeighted(frame, 0.2, canvas, 0.8, 0)

    # Show the output
    cv2.imshow("Hand Drawing", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
