import cv2
import mediapipe as mp
import pyttsx3

# Initialize pyttsx3 for voice output
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
tip_ids = [4, 8, 12, 16, 20]

def count_fingers(hand_landmarks, hand_label):
    fingers = []
    
    # Thumb logic
    if hand_label == "Right":
        fingers.append(1 if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x else 0)
    else:
        fingers.append(1 if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0] - 1].x else 0)

    # Other fingers
    for i in range(1, 5):
        fingers.append(1 if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i] - 2].y else 0)

    return sum(fingers)

num1, num2, result, operation = 0, 0, 0, ''
operation_done = False

while True:
    success, image = cap.read()
    image = cv2.flip(image, 1)
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    total_fingers = []
    hand_labels = []
    
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[i].classification[0].label
            hand_labels.append(label)

            finger_count = count_fingers(hand_landmarks, label)
            total_fingers.append((label, finger_count))
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Assign numbers from left/right hand
    if len(total_fingers) == 1:
        if total_fingers[0][0] == "Left":
            num1 = total_fingers[0][1]
        else:
            num2 = total_fingers[0][1]
    elif len(total_fingers) == 2:
        for label, count in total_fingers:
            if label == "Left":
                num1 = count
            else:
                num2 = count

    # Gesture-based operation
    operation_detected = ''
    if len(total_fingers) == 1:
        _, f = total_fingers[0]
        if f == 2:
            operation_detected = '+'
        elif f == 1:
            operation_detected = '-'
        elif f == 3:
            operation_detected = '*'
        elif f == 5:
            operation_detected = '/'

    # Only perform operation once per gesture
    if operation_detected and not operation_done:
        operation = operation_detected
        if operation == '+':
            result = num1 + num2
        elif operation == '-':
            result = num1 - num2
        elif operation == '*':
            result = num1 * num2
        elif operation == '/':
            result = num1 / num2 if num2 != 0 else 'Err'

        speak(f"{num1} {operation} {num2} equals {result}")
        operation_done = True

    if not operation_detected:
        operation_done = False  # reset flag when gesture removed

    # Display
    cv2.putText(image, f'Num1: {num1}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2)
    cv2.putText(image, f'Num2: {num2}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2)
    cv2.putText(image, f'Gesture: {operation}', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(image, f'Result: {result}', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
    cv2.putText(image, f'Press ESC to Exit', (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Gesture Calculator", image)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
