import cv2
import numpy as np
import time
import PoseModule as pm
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("jumpingJacksDemo.mp4")
detector = pm.poseDetector()
count = 0
dir = 0         # direction. 0 when going up and 1 when going down. a full repetition is doing these 2 both
pTime = 0
while True:
    success, img = cap.read()       #reading the camera/video feed
    img = cv2.flip(img,1)           # mirroring the image using flip
    img = cv2.resize(img, (1280, 720))      #making the image bigger so its easier to see
    img = detector.findPose(img,False)              # launching the pose detection model
    lmList = detector.findPosition(img, draw=False)         # extracting the key points
    if len(lmList) != 0:

        angle_r = detector.findAngle(img, 24, 12, 16)           #finding the angles between the key points
        angle_l = detector.findAngle(img, 15, 11, 23)
        angle_l_d = detector.findAngle(img, 27, 23, 28)
        angle = (angle_l+angle_r+angle_l_d)/3                   # averaging the angles to get a single angle to compare

        per = np.interp(angle, (14, 130), (0, 100))             # using interpolating, we set the value of the angle achieved in a percentage to compare
        bar = np.interp(angle, (14, 130), (650, 100))           # using interpolation, we draw a purple rectangle to denote progress
        print(angle, per)       # printing the angles and the percentage for testing purposes

        color = (255, 0, 255)

        #if the percentage reached 100% the progress bar will turn green and if the direction of the movement is up we increment and we set the direction down
        if per == 100:
            color = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1
        # same process but in the opposite direction to denote a full movement
        if per == 0:
            color = (0, 255, 0)
            if dir == 1:
                count += 0.5
                dir = 0

        # Draw Progress Bar
        cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4,color, 4)
        cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15,(255, 0, 0), 25)
    cTime = time.time()     # calculating the FPS for testing purposes
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5,
                (255, 0, 0), 5)
    cv2.imshow("Image", img)
    cv2.waitKey(1)