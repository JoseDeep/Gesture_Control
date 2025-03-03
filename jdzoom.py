import cv2 
import mediapipe as mp 
from math import hypot 
import numpy as np 
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def select_image():
    Tk().withdraw()
    filename = askopenfilename()
    return filename

mpHands = mp.solutions.hands 
hands = mpHands.Hands(
    static_image_mode=False, 
    model_complexity=1, 
    min_detection_confidence=0.75, 
    min_tracking_confidence=0.75, 
    max_num_hands=2
) 
Draw = mp.solutions.drawing_utils 

image_path = select_image()
if not image_path:
    print("No image selected.")
    exit()

image = cv2.imread(image_path)
if image is None:
    print("Failed to load image.")
    exit()

image_height, image_width, _ = image.shape

cap = cv2.VideoCapture(0) 

while True: 
    _, frame = cap.read() 
    frame = cv2.flip(frame, 1) 
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    Process = hands.process(frameRGB) 

    landmarkList = [] 
    if Process.multi_hand_landmarks: 
        for handlm in Process.multi_hand_landmarks: 
            for _id, landmarks in enumerate(handlm.landmark): 
                x, y = int(landmarks.x * frame.shape[1]), int(landmarks.y * frame.shape[0]) 
                landmarkList.append([_id, x, y]) 
            Draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS) 

    if landmarkList != []: 
        x_1, y_1 = landmarkList[4][1], landmarkList[4][2] 
        x_2, y_2 = landmarkList[8][1], landmarkList[8][2] 

        cv2.circle(frame, (x_1, y_1), 7, (0, 255, 0), cv2.FILLED) 
        cv2.circle(frame, (x_2, y_2), 7, (0, 255, 0), cv2.FILLED) 
        cv2.line(frame, (x_1, y_1), (x_2, y_2), (0, 255, 0), 3) 

        L = hypot(x_2 - x_1, y_2 - y_1) 
        zoom_level = np.interp(L, [15, 220], [1, 3]) 
        center_x, center_y = (x_1 + x_2) // 2, (y_1 + y_2) // 2 
        roi_size = int(min(image_width, image_height) / zoom_level) 
        top_left_x = max(center_x - roi_size // 2, 0)
        top_left_y = max(center_y - roi_size // 2, 0)
        bottom_right_x = min(center_x + roi_size // 2, image_width)
        bottom_right_y = min(center_y + roi_size // 2, image_height)
        roi = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        zoomed_image = cv2.resize(roi, (frame.shape[1], frame.shape[0]))
    else:
        zoomed_image = cv2.resize(image, (frame.shape[1], frame.shape[0]))

    combined_frame = np.hstack((frame, zoomed_image))
    cv2.imshow('Camera Feed and Zoomed Image', combined_frame)

    if cv2.waitKey(1) & 0xff == ord('q'): 
        break 

cap.release()
cv2.destroyAllWindows()
