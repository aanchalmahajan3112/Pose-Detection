import cv2
import mediapipe as mp     #library by Google for building various applications               
import numpy as np
mPose= mp.solutions.pose    #Pose class is used for pose estimation
mpDraw=mp.solutions.drawing_utils    #DrawingUtils is used to draw the landmarks on the image.
pose=mPose.Pose()

cap=cv2.VideoCapture('1.mp4')          # initializes a video capture object
drawspec1 = mpDraw.DrawingSpec(thickness=2,circle_radius=3,color=(0,0,255))   #determine appearance of landmarks on image
drawspec2 = mpDraw.DrawingSpec(thickness=4,circle_radius=5,color=(0,255,0))   #determine appearance of connections on image  

#while loop reads frames from the video capture object in a continuous manner
while True:
    success,img=cap.read()
    img=cv2.resize(img,(600,700))
    results=pose.process(img)   #processes image to obtain the pose landmarks
    mpDraw.draw_landmarks(img,results.pose_landmarks,mPose.POSE_CONNECTIONS,drawspec1,drawspec2)   #landmarks drawn on img being processed

    h,w,c= img.shape
    imgBlank=np.zeros([h,w,c])      #creating a blank image
    imgBlank.fill(255)               #filling the blank image with white color 
    mpDraw.draw_landmarks(imgBlank, results.pose_landmarks, mPose.POSE_CONNECTIONS, drawspec1, drawspec2)  #drawing landmarks on blank img


    cv2.imshow('pose_detection', img)
    cv2.imshow('ExtractedPose', imgBlank)
    cv2.waitKey(1)
