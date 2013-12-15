#!/usr/bin/python

from __future__ import division

import sys
import cv,cv2
import numpy as np
import time
import math

from helper import *
import svm


 
eyeBoxSize = 70 


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier('haarcascade_eye.xml')

minThreshold = 79;
maxThreshold = 255;
def setMinThreshold(args):
    global minThreshold
    minThreshold = args
def setMaxThreshold(args):
    global maxThreshold
    maxThreshold = args

pupilThreshold = 65;
glintThreshold = 240;
def setPupilThreshold(args):
    global pupilThreshold
    pupilThreshold = args
def setGlintThreshold(args):
    global glintThreshold
    glintThreshold = args

cv2.namedWindow('diff')
cv2.namedWindow('stereo')
cv.CreateTrackbar('min threshold', 'diff', minThreshold, 255, setMinThreshold) 
cv.CreateTrackbar('max threshold', 'diff', maxThreshold, 255, setMaxThreshold) 
cv.CreateTrackbar('pupil threshold', 'stereo', pupilThreshold, 255, setPupilThreshold) 
cv.CreateTrackbar('glint threshold', 'stereo', glintThreshold, 255, setGlintThreshold) 

#cams = [ cv2.VideoCapture(1) ]
cams = [ cv2.VideoCapture(1), cv2.VideoCapture(0) ]

for c in cams:
    c.set(cv.CV_CAP_PROP_FPS, 30)
    c.set(cv.CV_CAP_PROP_MODE, 4)

rows = 5
cols = 5
idx = 0
while True:
    if idx >= rows*cols:
        idx = 0
    err = False

    err, past, s1 = grabImgs(cams)
    err, imgs, s2 = grabImgs(cams)
    
    if s1[0] & 0x20:
        light, dark = imgs, past
    else:
        light, dark = past, imgs

    if len(past) == len(imgs):
        both = zip(past, imgs)
        diffs = []
        for s in both:
            diffs.append(cv2.absdiff(s[0], s[1]))
        diffStereo = np.concatenate(diffs, axis=1)
        im = np.concatenate(light, axis=1)
        im2 = np.concatenate(dark, axis=1)
      
        diffThresh = cv2.inRange(diffStereo, minThreshold, maxThreshold)
        diffThresh = cv2.dilate(diffThresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)))
        diffCont = diffThresh.copy()
        #diffThresh = cv2.adaptiveThreshold(diffStereo, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, maxThreshold)
        contours, hier = cv2.findContours(diffCont, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = filter(contourFilterFunc, contours)

        imCont = np.zeros((diffCont.shape[0], diffCont.shape[1], 3), dtype=np.uint8)
        cv2.drawContours(imCont, contours, -1, (255, 255, 255))

        imboth = np.concatenate((im, im2), axis=0)
        imboth = cv2.cvtColor(imboth, cv.CV_GRAY2BGR)

        eyeContours = np.zeros([2*int(eyeBoxSize/4), 0])
        
        PGVec = None
        a = [1991.84497391609,          179.411657101141,          54.3120306665886,         -10.0490343626226]
        b = [866.678760695169,          12.4457927037187,          1.41920775842972,         -15.9284542907887]
        #a = [1181.27159334425,         -27.8170122339234,         -211.926175603169,          13.5983541740038]
        #b = [3243.55717250692,         -34.9302611083195,          -522.08544499765,          25.1726262422174]

        mainPupil = None
        mainGlint = None
        minDist = sys.maxint
        mainBox = None
        for c in contours:
            (pos, eye) = getEye(im, c, eyeBoxSize)
            if eye != None:
                x, y = int(pos[0]), int(pos[1])
                half = int(eyeBoxSize/2)
                label = np.dot(svm.w, svm.scale * (np.reshape(eye, eyeBoxSize*eyeBoxSize, 'F') + svm.shift)) + svm.bias
                if label <= 0:
                    pupil, pCont = getCircle(eye, pupilThreshold)

                    #cv2.rectangle(imboth, (x-half, y-half), (x+half, y+half), (255, 255, 255), 2)
                    if pupil != None:
                        darkEye = im2[y-half+int(pupil[1])-int(eyeBoxSize/4):y-half+int(pupil[1])+int(eyeBoxSize/4), x-half+int(pupil[0])-int(eyeBoxSize/4):x-half+int(pupil[0])+int(eyeBoxSize/4)]
                        if darkEye.size != math.pow(2*int(eyeBoxSize/4),2):
                            continue
                        glint, gCont = getGlint(darkEye, pupil, glintThreshold, eyeBoxSize)

                        imGlint = np.zeros([2*int(eyeBoxSize/4), 2*int(eyeBoxSize/4)])
                        cv2.drawContours(imGlint, gCont, -1, (255, 255, 255))
                        eyeContours = np.concatenate([eyeContours, imGlint], axis=1)

                        #cv2.circle(imboth, (x-half+int(pupil[0]), y-half+int(pupil[1])), 7, (0, 255, 0))
                        dist = math.sqrt(math.pow(360 - (x-half+int(pupil[0])), 2) + math.pow(240 - (y-half+int(pupil[1])), 2))
                        if dist < minDist:
                            if glint != None:
                                mainPupil = (x-half+pupil[0], y-half+pupil[1])
                                mainGlint = (x-int(eyeBoxSize/4)+glint[0], y-int(eyeBoxSize/4)+glint[1])
                                mainBox = (x-half, y-half), (x+half, y+half)
                                #cv2.circle(imboth, (x-int(eyeBoxSize/4)+int(glint[0]), y-int(eyeBoxSize/4)+int(glint[1])), 3, (0, 0, 255))
        
        if mainPupil != None and mainGlint != None:
            PGVec = (mainPupil[0] - mainGlint[0], mainPupil[1] - mainGlint[1])
            cv2.circle(imboth, (int(mainPupil[0]), int(mainPupil[1])), 7, (0, 255, 0))
            cv2.circle(imboth, (int(mainGlint[0]), int(mainGlint[1])), 3, (0, 0, 255))
            cv2.rectangle(imboth, mainBox[0], mainBox[1], (255, 255, 255), 2)

        calib, coord = makeCalibrationImage(1920, 1080, rows, cols, idx)
        if PGVec != None:
            xg = a[0] + a[1]*PGVec[0] + a[2]*PGVec[1] + a[3]*PGVec[0]*PGVec[1]
            yg = b[0] + b[1]*PGVec[0] + b[2]*PGVec[1] + b[3]*PGVec[1]*PGVec[1]
            cv2.circle(calib, (int(xg), int(yg)), 3, (255, 255, 255), -1)
        cv2.imshow('calib', calib)
        #cv2.imshow('imCont', imCont)
        cv2.imshow('diff', diffThresh)
   
        cv2.imshow('stereo', imboth)
        if eyeContours.shape[1] != 0:
            cv2.imshow('eyes', eyeContours)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' '):
            if PGVec != None:
                print PGVec[0], PGVec[1], coord[0], coord[1]
        if key == ord('n'):
            idx += 1
    
    past = imgs

