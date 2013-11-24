#!/usr/bin/python

from __future__ import division

import cv,cv2
import numpy as np
import time
import itertools
import math


def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.izip_longest(fillvalue=fillvalue, *args)


def grabImgs(cams):
    imgs = []
    states = []
    success = True
    for c in cams:
        ret, f = c.read()

        if not ret:
            success = False
            break

        states.append(f[0][0])
        f = cv2.flip(f, 1)

        #grey = f.copy()
        #f = cv2.cvtColor(f, cv.CV_GRAY2BGR)
        #grey = cv2.equalizeHist(grey)

        """
        faces = face_cascade.detectMultiScale(grey, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)
        for (x,y,w,h) in faces:
            cv2.rectangle(f, (x,y), (x+w,y+h), (200,150,0), 2)
        """

        imgs.append(f)

    return (success, imgs, states)




face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier('haarcascade_eye.xml')


minThreshold = 30;
maxThreshold = 255;
def setMinThreshold(args):
    global minThreshold
    minThreshold = args
def setMaxThreshold(args):
    global maxThreshold
    maxThreshold = args

cv2.namedWindow('diff')
cv2.namedWindow('roi')
cv.CreateTrackbar('min threshold', 'diff', minThreshold, 255, setMinThreshold) 
cv.CreateTrackbar('max threshold', 'diff', maxThreshold, 255, setMaxThreshold) 

#cams = [ cv2.VideoCapture(1) ]
cams = [ cv2.VideoCapture(1), cv2.VideoCapture(0) ]

for c in cams:
    c.set(cv.CV_CAP_PROP_FPS, 30)
    c.set(cv.CV_CAP_PROP_MODE, 4)

while True:
    err = False

    err, imgs, s1 = grabImgs(cams)
    err, past, s2 = grabImgs(cams)
    
    if s1[0] & 0x20:
        light, dark = past, imgs
    else:
        light, dark = imgs, past

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

        def filterFunc(c):
            hull = cv2.convexHull(c)
            area = cv2.contourArea(c)
            hullArea = cv2.contourArea(hull)
            return area == 0 or hullArea/area <= 1.3

        contours = filter(filterFunc, contours)
        eyes = []

        imCont = np.zeros((diffCont.shape[0], diffCont.shape[1], 3), dtype=np.uint8)
        cv2.drawContours(imCont, contours, -1, (255, 255, 255))

        cv2.imshow('imCont', imCont)
        cv2.imshow('diff', diffThresh)
   
        cv2.imshow('stereo', imboth)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        eyeBoxSize = 30
        for c in contours:
            moments = cv2.moments(c)
            if moments['m00'] > 0:
                x, y = moments['m10']/moments['m00'], moments['m01']/moments['m00']
                half = int(eyeBoxSize/2)
                roi = im2[y-half:y+half, x-half:x+half]
                if roi.size == eyeBoxSize*eyeBoxSize:
                    eyes.append(roi.copy())

        if len(eyes) > 0:
            sidey = int(math.ceil(math.sqrt(len(eyes))))
            sidex = int(math.floor(math.sqrt(len(eyes)) + 0.5))
            fill = sidex*sidey - len(eyes)
            
            clicked = np.zeros((sidey, sidex), dtype=np.uint8)
            def onClick(event, x, y, flags, param):
                ix = math.floor(x/eyeBoxSize)
                iy = math.floor(y/eyeBoxSize)
                if event == cv2.EVENT_FLAG_LBUTTON:
                    clicked[iy][ix] = 1 
                elif event == cv2.EVENT_FLAG_RBUTTON:
                    clicked[iy][ix] = 0 

            allEyeImgs = list(grouper(sidex, eyes, np.zeros((eyeBoxSize, eyeBoxSize), dtype=np.uint8)))

            alleyes = np.concatenate([np.concatenate(row, axis=1) for row in allEyeImgs], axis=0)
            alleyes = cv2.cvtColor(alleyes, cv.CV_GRAY2BGR)

            cv2.setMouseCallback('roi', onClick)
        
            while True:
                displayAllEyes = alleyes.copy()
                for y in xrange(sidey):
                    for x in xrange(sidex):
                        if clicked[y][x] == 1:
                            cv2.rectangle(displayAllEyes, (x*eyeBoxSize, y*eyeBoxSize), ((x+1)*eyeBoxSize, (y+1)*eyeBoxSize), (255, 0, 0), 2)
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
                            else:
                                cv2.imwrite('SVM_Data/neg/%s-N%03d.png' % (tstr, numNeg), allEyeImgs[y][x])
                                numNeg += 1

                    break

                elif innerKey == ord('q'):
                    break
       
        

    if key == ord('q'):
        break
    
    past = imgs

