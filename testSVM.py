#!/usr/bin/python

from __future__ import division

import cv,cv2
import numpy as np
from numpy import linalg as LA
import time
import math

import config
import eyes
from helper import *
import svm


cv2.namedWindow('contours')
cv2.namedWindow('stereo')
cv2.namedWindow('roi')
cv.CreateTrackbar('pupilThresh1Min', 'contours', config.pupilThresh1Min, 255, config.setPupilThresh1Min) 
cv.CreateTrackbar('pupilThresh1Max', 'contours', config.pupilThresh1Max, 255, config.setPupilThresh1Max) 
cv.CreateTrackbar('pupil threshold', 'stereo',   config.pupilThresh2Min, 255, config.setPupilThresh2Min) 
cv.CreateTrackbar('glint threshold', 'stereo',   config.glintThreshMin,  255, config.setGlintThreshMin)

#cams = [ cv2.VideoCapture(1), cv2.VideoCapture(0) ]
cams = [ cv2.VideoCapture(1) ]

for c in cams:
    c.set(cv.CV_CAP_PROP_FPS, 30)
    c.set(cv.CV_CAP_PROP_MODE, 4)

svmCorrect = 0
svmError = 0
while True:
    light, dark = grabLightDarkPair(cams)

    if light is not None and dark is not None:
        imboth = np.concatenate((np.concatenate(light, axis=1), np.concatenate(dark, axis=1)), axis=0)
        imboth = cv2.cvtColor(imboth, cv.CV_GRAY2BGR)

        contours = eyes.findBlobs(light, dark)
        contours = contours[0]

        half = int(config.eyePatchSize/2)
        for c in contours:
            (pos, eye) = eyes.getEyePatch(light[0], c)
            if eye != None:
                pos = (int(pos[0]), int(pos[1]))
                label = kernel(svm.w, svm.scale * (np.reshape(eye, config.eyePatchSize*config.eyePatchSize, 'F') + svm.shift)) + svm.bias
                if label <= 0:
                    cv2.rectangle(imboth, (pos[0]-half, pos[1]-half), (pos[0]+half, pos[1]+half), (255, 0, 0), 2)

        cv2.imshow('stereo', imboth)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        eyeList = []
        for c in contours:
            (pos, eye) = eyes.getEyePatch(light[0], c)
            if eye != None:
                eyeList.append(eye)

        if len(eyeList) > 0:
            sidey = int(math.ceil(math.sqrt(len(eyeList))))
            sidex = int(math.floor(math.sqrt(len(eyeList)) + 0.5))
            fill = sidex*sidey - len(eyeList)
            
            clicked = np.zeros((sidey, sidex), dtype=np.int8)

            def onClick(event, x, y, flags, param):
                ix = math.floor(x/config.eyePatchSize)
                iy = math.floor(y/config.eyePatchSize)
                if event == cv2.EVENT_FLAG_LBUTTON:
                    if clicked[iy][ix] < 1:
                        clicked[iy][ix] += 1
                elif event == cv2.EVENT_FLAG_RBUTTON:
                    if clicked[iy][ix] > -1:
                        clicked[iy][ix] += -1

            allEyeImgs = list(grouper(sidex, eyeList, np.zeros((config.eyePatchSize, config.eyePatchSize), dtype=np.uint8)))

            alleyes = np.concatenate([np.concatenate(row, axis=1) for row in allEyeImgs], axis=0)
            alleyes = cv2.cvtColor(alleyes, cv.CV_GRAY2BGR)

            for y in xrange(sidey):
                for x in xrange(sidex):
                    if np.amin(allEyeImgs[y][x]) == np.amax(allEyeImgs[y][x]):
                        clicked[y][x] = 0
                        continue
                    label = kernel(svm.w, svm.scale * (np.reshape(allEyeImgs[y][x], config.eyePatchSize*config.eyePatchSize, 'F') + svm.shift)) + svm.bias
                    if label < 0:
                        clicked[y][x] = 1
                    else:
                        clicked[y][x] = -1

            cv2.setMouseCallback('roi', onClick)
        
            while True:
                displayAllEyes = alleyes.copy()
                for y in xrange(sidey):
                    for x in xrange(sidex):
                        if clicked[y][x] > 0:
                            cv2.rectangle(displayAllEyes, (x*config.eyePatchSize, y*config.eyePatchSize), ((x+1)*config.eyePatchSize-1, (y+1)*config.eyePatchSize-1), (0, 255, 0), 1)
                        #elif clicked[y][x] == 0:
                        #    cv2.rectangle(displayAllEyes, (x*config.eyePatchSize, y*config.eyePatchSize), ((x+1)*config.eyePatchSize-1, (y+1)*config.eyePatchSize-1), (255, 0, 0), 1)
                        elif clicked[y][x] < 0:
                            cv2.rectangle(displayAllEyes, (x*config.eyePatchSize, y*config.eyePatchSize), ((x+1)*config.eyePatchSize-1, (y+1)*config.eyePatchSize-1), (0, 0, 255), 1)
                cv2.imshow('roi', displayAllEyes)
                innerKey = cv2.waitKey(1) & 0xFF 
                if innerKey == ord(' '):
                    tstr = time.strftime('%Y%m%dT%H%M%S', time.gmtime())

                    for y in xrange(sidey):
                        for x in xrange(sidex):
                            if np.amin(allEyeImgs[y][x]) == np.amax(allEyeImgs[y][x]):
                                continue
                            elif clicked[y][x] == 0:
                                svmError += 1
                            elif clicked[y][x] == 1 or clicked[y][x] == -1:
                                svmCorrect += 1 
                    print svmCorrect, svmError

                    break

                elif innerKey == ord('q'):
                    break
        

    if key == ord('q'):
        break

