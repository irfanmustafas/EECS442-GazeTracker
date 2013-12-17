#!/usr/bin/python

from __future__ import division

import sys
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
#cv.CreateTrackbar('pupil threshold', 'stereo',   config.pupilThresh2Min, 255, config.setPupilThresh2Min) 
#cv.CreateTrackbar('glint threshold', 'stereo',   config.glintThreshMin,  255, config.setGlintThreshMin)

#cams = [ cv2.VideoCapture(1), cv2.VideoCapture(0) ]
cams = [ cv2.VideoCapture(1) ]

for c in cams:
    c.set(cv.CV_CAP_PROP_FPS, 30)
    c.set(cv.CV_CAP_PROP_MODE, 4)

idx = 0
while True:
    if idx >= config.calibRows * config.calibCols:
        idx = 0

    light, dark = grabLightDarkPair(cams)

    if light is not None and dark is not None:
        imboth = np.concatenate((np.concatenate(light, axis=1), np.concatenate(dark, axis=1)), axis=0)
        imboth = cv2.cvtColor(imboth, cv.CV_GRAY2BGR)

        contours = eyes.findBlobs(light, dark)
        pupilGlintVectors = eyes.findEyes(light, dark, contours, imboth)

        a = [1438.8497483032,          245.754013157552,          21.1839369576548,          16.8707657712166]
        b = [1480.93392967789,         -2.47110909538142,          326.671102334977,          20.7502568126427]

        calib, coord = makeCalibrationImage(1920, 1080, idx)
        for cam in range(len(cams)):
            PGVecs = pupilGlintVectors[cam]
            for PGVec in PGVecs:
                xg = a[0] + a[1]*PGVec[0] + a[2]*PGVec[1] + a[3]*PGVec[0]*PGVec[1]
                yg = b[0] + b[1]*PGVec[0] + b[2]*PGVec[1] + b[3]*PGVec[1]*PGVec[1]
                cv2.circle(calib, (int(xg), int(yg)), 3, (255*cam, 255, 255), -1)

        cv2.imshow('calib', calib)
        cv2.imshow('stereo', imboth)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' '):
            pgv = pupilGlintVectors[0]
            if len(pgv) != 0:
                print pgv[0][0], pgv[0][1], coord[0], coord[1]
        if key == ord('n'):
            idx += 1

