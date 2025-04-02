import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

def normalize_landmarks(landmarks):
    """Normalize hand landmarks relative to the wrist (landmark 0)."""
    wrist = landmarks[0]  # Reference point (wrist)
    landmarks = np.array(landmarks) - wrist  # Shift origin to wrist

    # Scale to fit in [-1, 1]
    max_value = np.max(np.abs(landmarks))
    if max_value > 0:
        landmarks /= max_value

    return landmarks.flatten()  # Flatten for easier storage

def get_finger_states(landmarks) :
    """
    Determines which fingers are open (1) or closed (0) based on hand landmarks.

    Returns:
    - A list of 5 values [thumb, index, middle, ring, little],
      where 1 means open and 0 means closed.
    """
    landmarks = np.array(landmarks).reshape((21,3))
    finger_states = []

    # Determining which hand it is 

    if landmarks[17][0] < landmarks[5][0] :
        hand_type = "right"
    else :
        hand_type = "left"

    if hand_type == "right" :
        if landmarks[4][0] > landmarks[2][0] :
            finger_states.append(1)
        else :
            finger_states.append(0)
    else :
        if landmarks[4][0] < landmarks[2][0] :
            finger_states.append(1)
        else :
            finger_states.append(0)

    # Other fingers : compare y- coordinate 
    fingers = [(8,6),(12,10),(16,14),(20,18)]

    for tip,base in fingers :
        if landmarks[tip][1] < landmarks[base][1] :
            finger_states.append(1)
        else :
            finger_states.append(0)
    return finger_states

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            normalized_landmarks = normalize_landmarks(landmarks)
            
            # print("Normalized Landmarks:", normalized_landmarks)  # Should print stable values
            finger_states = get_finger_states(landmarks)

            if all(x == 0 for x in finger_states):
                gesture_text = "Fist 🤜"
            elif all(x == 1 for x in finger_states):
                gesture_text = "Open Hand ✋"
            elif finger_states == [0,1,1,0,0] :
                gesture_text = "Victory Sign ✌️"
            elif finger_states == [1,1,0,0,1] :
                gesture_text = "Yo Yo Sign 🤟"
            else:
                gesture_text = f"Gesture: {finger_states}"

            if gesture_text :
                cv2.putText(frame , gesture_text , org = (50,50) ,
                             fontFace = cv2.FONT_HERSHEY_COMPLEX , fontScale=1 ,
                              color = (0,255,0) ,thickness=2 )
            
    cv2.imshow("Hand Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
