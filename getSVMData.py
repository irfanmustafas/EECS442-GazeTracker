#!/usr/bin/python

from __future__ import division

import cv,cv2
import numpy as np
from numpy import linalg as LA
import time
import math

from helper import *
import svm

#from svmcv import SVMCV


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier('haarcascade_eye.xml')

eyeBoxSize = 70

minThreshold = 79;
maxThreshold = 255;
def setMinThreshold(args):
    global minThreshold
    minThreshold = args
def setMaxThreshold(args):
    global maxThreshold
    maxThreshold = args

cv2.namedWindow('imCont')
cv2.namedWindow('roi')
cv.CreateTrackbar('min threshold', 'imCont', minThreshold, 255, setMinThreshold) 
cv.CreateTrackbar('max threshold', 'imCont', maxThreshold, 255, setMaxThreshold) 

#cams = [ cv2.VideoCapture(1) ]
cams = [ cv2.VideoCapture(1), cv2.VideoCapture(0) ]

for c in cams:
    c.set(cv.CV_CAP_PROP_FPS, 30)
    c.set(cv.CV_CAP_PROP_MODE, 4)

while True:
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
      
        imboth = np.concatenate((im, im2), axis=0)

        diffThresh = cv2.inRange(diffStereo, minThreshold, maxThreshold)
        diffThresh = cv2.dilate(diffThresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)))
        diffCont = diffThresh.copy()
        #diffThresh = cv2.adaptiveThreshold(diffStereo, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, maxThreshold)
        contours, hier = cv2.findContours(diffCont, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        contours = filter(contourFilterFunc, contours)

        imCont = np.zeros((diffCont.shape[0], diffCont.shape[1], 3), dtype=np.uint8)
        cv2.drawContours(imCont, contours, -1, (255, 255, 255))

        half = int(eyeBoxSize/2)
        for c in contours:
            (pos, eye) = getEye(im, c, eyeBoxSize)
            if eye != None:
                label = kernel(svm.w, svm.scale * (np.reshape(eye, eyeBoxSize*eyeBoxSize, 'F') + svm.shift)) + svm.bias
                if label <= 0:
                    cv2.rectangle(imboth, (pos[0]-half, pos[1]-half), (pos[0]+half, pos[1]+half), (255, 0, 0), 2)

        cv2.imshow('imCont', imCont)
        cv2.imshow('diff', diffThresh)
   
        cv2.imshow('stereo', imboth)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        eyes = []
        for c in contours:
            (pos, eye) = getEye(im, c, eyeBoxSize)
            if eye != None:
                eyes.append(eye)

        if len(eyes) > 0:
            sidey = int(math.ceil(math.sqrt(len(eyes))))
            sidex = int(math.floor(math.sqrt(len(eyes)) + 0.5))
            fill = sidex*sidey - len(eyes)
            
            clicked = np.zeros((sidey, sidex), dtype=np.int8)

            def onClick(event, x, y, flags, param):
                ix = math.floor(x/eyeBoxSize)
                iy = math.floor(y/eyeBoxSize)
                if event == cv2.EVENT_FLAG_LBUTTON:
                    if clicked[iy][ix] < 1:
                        clicked[iy][ix] += 1
                elif event == cv2.EVENT_FLAG_RBUTTON:
                    if clicked[iy][ix] > -1:
                        clicked[iy][ix] += -1

            allEyeImgs = list(grouper(sidex, eyes, np.zeros((eyeBoxSize, eyeBoxSize), dtype=np.uint8)))

            alleyes = np.concatenate([np.concatenate(row, axis=1) for row in allEyeImgs], axis=0)
            alleyes = cv2.cvtColor(alleyes, cv.CV_GRAY2BGR)

            for y in xrange(sidey):
                for x in xrange(sidex):
                    if np.amin(allEyeImgs[y][x]) == np.amax(allEyeImgs[y][x]):
                        clicked[y][x] = -1
                        continue
                    label = kernel(svm.w, svm.scale * (np.reshape(allEyeImgs[y][x], eyeBoxSize*eyeBoxSize, 'F') + svm.shift)) + svm.bias
                    if label < 0:
                        clicked[y][x] = 1

            cv2.setMouseCallback('roi', onClick)
        
            while True:
                displayAllEyes = alleyes.copy()
                for y in xrange(sidey):
                    for x in xrange(sidex):
                        if clicked[y][x] > 0:
                            cv2.rectangle(displayAllEyes, (x*eyeBoxSize, y*eyeBoxSize), ((x+1)*eyeBoxSize-1, (y+1)*eyeBoxSize-1), (0, 255, 0), 1)
                        elif clicked[y][x] == 0:
                            cv2.rectangle(displayAllEyes, (x*eyeBoxSize, y*eyeBoxSize), ((x+1)*eyeBoxSize-1, (y+1)*eyeBoxSize-1), (255, 0, 0), 1)
                        elif clicked[y][x] < 0:
                            cv2.rectangle(displayAllEyes, (x*eyeBoxSize, y*eyeBoxSize), ((x+1)*eyeBoxSize-1, (y+1)*eyeBoxSize-1), (0, 0, 255), 1)
                cv2.imshow('roi', displayAllEyes)
                innerKey = cv2.waitKey(1) & 0xFF 
                if innerKey == ord(' '):
                    numPos = 0
                    numNeg = 0
                    tstr = time.strftime('%Y%m%dT%H%M%S', time.gmtime())

                    for y in xrange(sidey):
                        for x in xrange(sidex):
                            if clicked[y][x] == 1:
                                cv2.imwrite('SVM_Data/pos/%s-P%03d.png' % (tstr, numPos), allEyeImgs[y][x])
                                numPos += 1 
                            elif clicked[y][x] == 0:
                                cv2.imwrite('SVM_Data/neg/%s-N%03d.png' % (tstr, numNeg), allEyeImgs[y][x])
                                numNeg += 1

                    break

                elif innerKey == ord('q'):
                    break
       
        

    if key == ord('q'):
        break

