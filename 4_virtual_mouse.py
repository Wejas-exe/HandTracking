import cv2
import mediapipe as mp
import numpy as np
import pyautogui

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence =0.7 , min_tracking_confidence = 0.7) 

screen_width , screen_height = pyautogui.size()
dragging = False

cap = cv2.VideoCapture(0)

def get_finger_states(landmarks) :

    landmarks = np.array(landmarks).reshape((21,3))
    finger_states = []

    hand_type = "right" if landmarks[17][0] < landmarks[5][0] else "left"

    if hand_type == "right" :
        if(landmarks[4][0] > landmarks[2][0]) :
            finger_states.append(1)
        else :
            finger_states.append(0)
    else :
        finger_states.append(1 if landmarks[4][0] < landmarks[2][0] else 0)
    
    fingers = [(8,6),(12,10),(16,14),(20,18)] 
    for tip,base in fingers :
        finger_states.append(1 if landmarks[tip][1] < landmarks[base][1] else 0)

    return finger_states


while cap.isOpened() :
    ret , frame = cap.read()
    if not ret :
        break

    frame = cv2.flip(frame ,1)

    rgb_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks :
        for hand_landmarks in results.multi_hand_landmarks :
            mp_draw.draw_landmarks(frame , hand_landmarks , mp_hands.HAND_CONNECTIONS)

        landmarks = [[lm.x ,lm.y, lm.z] for lm in hand_landmarks.landmark]
        finger_states = get_finger_states(landmarks)

        index_x , index_y = landmarks[8][0] , landmarks[8][1]

        cursor_x = int(index_x * screen_width)
        cursor_y = int(index_y * screen_height)

        pyautogui.moveTo(cursor_x , cursor_y , duration= 0.1)
        
        # Left Click (Only Index Finger Up)
        if finger_states == [0, 1, 0, 0, 0]:
            pyautogui.click()
            cv2.putText(frame, "Click", (50, 50),
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Right Click (Only Middle Finger Up)
        elif finger_states == [0, 0, 1, 0, 0]:
            pyautogui.rightClick()
            cv2.putText(frame, "Right Click", (50, 90),
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Drag Mode (Index & Middle Finger Up)
        elif finger_states == [0, 1, 1, 0, 0]:
            if not dragging:
                pyautogui.mouseDown()
                dragging = True
            cv2.putText(frame, "Dragging", (50, 130),
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            if dragging:
                pyautogui.mouseUp()
                dragging = False

        # Scroll Mode (Move Hand Up/Down While Dragging)
        if dragging:
            pyautogui.scroll(int((0.5 - index_y) * 20))  # Adjust scroll sensitivity
            
    cv2.imshow("Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()