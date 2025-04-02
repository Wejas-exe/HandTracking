import cv2 
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence = 0.5 ,
                           min_tracking_confidence = 0.5) as holistic:
    while cap.isOpened() :
        ret , frame = cap.read()
        if not ret :
            break
    
        rgb_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        mp_draw.draw_landmarks(frame , results.face_landmarks ,
                                mp_holistic.FACEMESH_TESSELATION)
        mp_draw.draw_landmarks(frame , results.pose_landmarks , 
                               mp_holistic.POSE_CONNECTIONS)
        mp_draw.draw_landmarks(frame , results.right_hand_landmarks , 
                               mp_holistic.HAND_CONNECTIONS)
        mp_draw.draw_landmarks(frame , results.left_hand_landmarks , 
                               mp_holistic.HAND_CONNECTIONS)
        cv2.imshow("webcam" , frame)

        if cv2.waitKey(1) == ord('q') :
            break

cap.release()
cv2.destroyAllWindows()