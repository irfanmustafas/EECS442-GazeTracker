#!/usr/bin/python

from __future__ import division

import cv,cv2
import numpy as np
import time
import math

import config
import eyes
from helper import *
import svm


cv2.namedWindow('contours')
cv2.namedWindow('stereo')
cv.CreateTrackbar('pupilThresh1Min', 'contours', config.pupilThresh1Min, 255, config.setPupilThresh1Min) 
cv.CreateTrackbar('pupilThresh1Max', 'contours', config.pupilThresh1Max, 255, config.setPupilThresh1Max) 
cv.CreateTrackbar('pupil threshold', 'stereo',   config.pupilThresh2Min, 255, config.setPupilThresh2Min) 
cv.CreateTrackbar('glint threshold', 'stereo',   config.glintThreshMin,  255, config.setGlintThreshMin)

cams = [ cv2.VideoCapture(1), cv2.VideoCapture(0) ]

for c in cams:
    c.set(cv.CV_CAP_PROP_FPS, 30)
    c.set(cv.CV_CAP_PROP_MODE, 4)

while True:
    light, dark = grabLightDarkPair(cams)

    if light is not None and dark is not None:
        imboth = np.concatenate((np.concatenate(light, axis=1), np.concatenate(dark, axis=1)), axis=0)
        imboth = cv2.cvtColor(imboth, cv.CV_GRAY2BGR)

        contours = eyes.findBlobs(light, dark)

        eyeContours = np.zeros([2*int(config.eyePatchSize/4), 0])

        for cam in range(len(light)):
            for c in contours[cam]:
                (pos, eye) = eyes.getEyePatch(light[cam], c)
                if eye != None:
                    x, y = pos[0], pos[1]
                    half = int(config.eyePatchSize / 2)
                    label = np.dot(svm.w, svm.scale * (np.reshape(eye, config.eyePatchSize*config.eyePatchSize, 'F') + svm.shift)) + svm.bias
                    if label <= 0:
                        pupil, pCont = eyes.getCircle(eye, config.pupilThresh2Min)

                        cv2.rectangle(imboth, (int(x-half), int(y-half)), (int(x+half), int(y+half)), (255, 255, 255), 2)
                        if pupil != None:
                            darkEye = dark[cam][y-half+pupil[1]-int(config.eyePatchSize/4):y-half+pupil[1]+int(config.eyePatchSize/4), x-half+pupil[0]-int(config.eyePatchSize/4):x-half+pupil[0]+int(config.eyePatchSize/4)]
                            if darkEye.size != math.pow(2*int(config.eyePatchSize/4),2):
                                continue
                            glint, gCont = eyes.getCircle(darkEye, config.glintThreshMin)

                            imGlint = np.zeros([2*int(config.eyePatchSize/4), 2*int(config.eyePatchSize/4)])
                            cv2.drawContours(imGlint, gCont, -1, (255, 255, 255))
                            eyeContours = np.concatenate([eyeContours, imGlint], axis=1)

                            cv2.circle(imboth, (int(x-half+pupil[0]), int(y-half+pupil[1])), 7, (0, 255, 0))
                            if glint != None:
                                cv2.circle(imboth, (int(x-int(config.eyePatchSize/4)+glint[0]), int(y-int(config.eyePatchSize/4)+glint[1])), 3, (0, 0, 255))
        
        cv2.imshow('stereo', imboth)
        if eyeContours.shape[1] != 0:
            cv2.imshow('eyes', eyeContours)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

