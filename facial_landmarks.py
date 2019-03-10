from imutils.video import VideoStream, FileVideoStream
from imutils import face_utils
import argparse
import imutils
import time
import cv2
import dlib
import numpy as np
from threading import Thread
from scipy.spatial import distance as dist
import sys
import cognitive_face as cf
import json

with open('faceapi.json') as file:
    json2 = json.load(file)
    key = json2['key']
    BASE_URL = json2['serviceUrl']
    group = json2['groupId']

cf.BaseUrl.set(BASE_URL)

try:
    e = cf.Key.set(key)
except:
    print( "Incorrect subscription key")
    sys.exit()


path="shape_predictor_68_face_landmarks.dat"
predictor=dlib.shape_predictor(path)
detector=dlib.get_frontal_face_detector()       

def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

def eye_close(image):
    path="shape_predictor_68_face_landmarks.dat"
    predictor=dlib.shape_predictor(path)
    detector=dlib.get_frontal_face_detector()

    EYE_AR_THRESH = 0.4

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    vs = FileVideoStream(image).start()
    time.sleep(2.0)
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
    return leftEAR, rightEAR

def get_landmarks(im):
    rects=detector(im,1)#image and no.of rectangles to be drawn
    if len(rects)>1:
        print("Toomanyfaces")
        return np.matrix([0])
    if len(rects)==0:
        print("Toofewfaces")
        return np.matrix([0])
    return np.matrix([[p.x,p.y] for p in predictor(im,rects[0]).parts()])  

def place_landmarks(im,landmarks):
    im=im.copy()
    for idx,point in enumerate(landmarks):
        pos=(point[0,0],point[0,1])
        cv2.putText(im,str(idx),pos,fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=0.3,color=(0,255,255))
        cv2.circle(im,pos,3,color=(0,255,255))
    return im 

def upper_lip(landmarks):
    top_lip=[]
    for i in range(50,53):
        top_lip.append(landmarks[i])
    for j in range(61,64):
        top_lip.append(landmarks[j])
    top_lip_point=(np.squeeze(np.asarray(top_lip)))
    top_mean=np.mean(top_lip_point,axis=0)
    return int(top_mean[1])
    
def low_lip(landmarks):
    lower_lip=[]
    for i in range(65,68):
        lower_lip.append(landmarks[i])
    for j in range(56,59):
        lower_lip.append(landmarks[j])
    lower_lip_point=(np.squeeze(np.asarray(lower_lip)))
    lower_mean=np.mean(lower_lip_point,axis=0)
    return int(lower_mean[1])
               
def decision(image):
    landmarks=get_landmarks(image)
    if(landmarks.all()==[0]):
        return -10#Dummy value to prevent error
    top_lip=upper_lip(landmarks)
    lower_lip=low_lip(landmarks)
    distance=abs(top_lip-lower_lip)
    return distance 

def mouth_open(image):  
    cap=cv2.VideoCapture(image)
    ret,frame=cap.read()
    if(ret==True):
        distance=decision(frame)
        if(distance>21): 
            ans = 'mouth_open'
            print(ans)
    else:
        pass
    cap.release()
    cv2.destroyAllWindows()

def yaw_and_roll(image):
    face_info = cf.face.detect(image, face_id=False, attributes='headPose')
    roll = face_info[0]['faceAttributes']['headPose']['roll']
    yaw = face_info[0]['faceAttributes']['headPose']['yaw']
    return roll, yaw