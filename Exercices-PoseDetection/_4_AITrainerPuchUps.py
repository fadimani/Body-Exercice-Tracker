import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose



count =0
position=None

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
    while 1:
        success, image = cap.read()
        image = cv2.resize(image, (1280, 720))
        image = cv2.flip(image, 1)
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)


        if results.pose_landmarks:
            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())



            #Adding the push up part
            lmList = []

            for id, im in enumerate(results.pose_landmarks.landmark):
                h, w,c = image.shape
                cx, cy = int(im.x * w), int(im.y * h)
                lmList.append([id,cx,cy])


            if ((lmList[12][2] - lmList[14][2])>=15 and (lmList[11][2] - lmList[13][2])>=15):
                position = "down"
                print(position)
                cv2.putText(image, position, (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,(0, 255, 0), 3)
            if ((lmList[12][2] - lmList[14][2])<=5 and (lmList[11][2] - lmList[13][2])<=5) and position == "down":
                position = "up"
                print(position)
                cv2.putText(image, position, (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,(0, 255, 0), 3)
                count +=1
                print(count)

        cv2.putText(image, str(count), (70, 100), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 0), 3)

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('Pose Test', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()