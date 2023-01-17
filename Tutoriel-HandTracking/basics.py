import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

# using the hand detecting algo
mpHands = mp.solutions.hands
hands = mpHands.Hands()

#helps us draw the points on the hands detected
mpDraw = mp.solutions.drawing_utils

# these are for drawing the fps
pTime = 0       #previous time
cTime = 0       #current time

while True:
    # reading input images from the camera
    success, img = cap.read()
    # converting them to rgb, cus opencv takes them in bgr for some fucking reason
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # putting the results of the hand detecting algo here
    results = hands.process(imgRGB)
    # this confirms if we detected a hand or not. if detected, it will print the landmark, if not, it says none
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        #   handLms is a single hand, this loops around the hands
        for handLms in results.multi_hand_landmarks:

            # mpHands.HAND_CONNECTIONS is what draws the connections
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # for this loop, for each hand, we will be gathering some info like id of the finger points and the landmark
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)  # id of each finger point and its landmark(location but in ratio, not in pixel)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)       # we multiply by the image size to get the pixel locations
                #print(id, cx, cy)
                if id == 8:     #this is a nice way to highlight a certain finger point
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                    print(id, cx, cy)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    #this is how we write stuff on the screen
    #           img,   text,         position,     font,            size, color,     the thickness of it oh ma gahd
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)  # this is like sleep, for 1ms, it's helpful in reducing a video frame rate
