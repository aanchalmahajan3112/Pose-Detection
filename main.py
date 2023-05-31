import cv2
import mediapipe as mp
import numpy as np
mPose= mp.solutions.pose
mpDraw=mp.solutions.drawing_utils
pose=mPose.Pose()

cap=cv2.VideoCapture('1.mp4')
drawspec1 = mpDraw.DrawingSpec(thickness=2,circle_radius=3,color=(0,0,255))
drawspec2 = mpDraw.DrawingSpec(thickness=4,circle_radius=5,color=(0,255,0))


while True:
    success,img=cap.read()
    img=cv2.resize(img,(600,700))
    results=pose.process(img)
    mpDraw.draw_landmarks(img,results.pose_landmarks,mPose.POSE_CONNECTIONS,drawspec1,drawspec2)

    h,w,c= img.shape
    imgBlank=np.zeros([h,w,c])
    imgBlank.fill(255)
    mpDraw.draw_landmarks(imgBlank, results.pose_landmarks, mPose.POSE_CONNECTIONS, drawspec1, drawspec2)


    cv2.imshow('pose_detection', img)
    cv2.imshow('ExtractedPose', imgBlank)
    cv2.waitKey(1)
