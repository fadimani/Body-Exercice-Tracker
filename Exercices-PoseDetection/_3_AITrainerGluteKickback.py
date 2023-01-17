import cv2
import numpy as np
import time
import PoseModule as pm
cap = cv2.VideoCapture(0)
detector = pm.poseDetector()
count = 0
dir = 0         # direction. 0 when going up and 1 when going down. a full curl is doing these 2 both
pTime = 0
while True:
    success, img = cap.read()
    img = cv2.flip(img,1)
    img = cv2.resize(img, (1280, 720))
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)
    if len(lmList) != 0:
        angle_left = detector.findAngle(img, 26, 24, 12)   # this must stay =90
        angle_right = detector.findAngle(img, 25, 23, 11)   # this must be 180
        angle_right2 = detector.findAngle(img, 27, 25, 23)   # this must 90

        angle = (angle_right2+angle_right)/2

        if(70<angle_left<120):
            per = np.interp(angle, (200, 230), (0, 100))
            bar = np.interp(angle, (200, 230), (650, 100))
            print(angle, per)
            color = (255, 0, 255)

            if per == 100:
                color = (0, 255, 0)
                if dir == 0:
                    count += 0.5
                    dir = 1
            if per == 0:
                color = (0, 255, 0)
                if dir == 1:
                    count += 0.5
                    dir = 0
            # Draw Bar
            cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
            cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
            cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4,color, 4)
            cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15,(255, 0, 0), 25)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5,
                (255, 0, 0), 5)
    cv2.imshow("Image", img)
    cv2.waitKey(1)