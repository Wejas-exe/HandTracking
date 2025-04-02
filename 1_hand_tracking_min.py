import cv2
import mediapipe as mp

# initialize mediapipe hands 
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence =0.5 ,min_tracking_confidence=0.5) as hands :
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB (required for MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks :
            for hand_landmarks in results.multi_hand_landmarks :
                mp_draw.draw_landmarks(frame , hand_landmarks , mp_hands.HAND_CONNECTIONS)
        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()