import cv,cv2
import numpy as np
import itertools
import sys
import math

import config

def rbf_kernel(a, b, sigma):
    print np.exp(-(1/(2*sigma*sigma))*(LA.norm(a - b)*LA.norm(a - b)))
    return np.exp(-(1/(2*sigma*sigma))*(LA.norm(a - b)*LA.norm(a - b)))

def kernel(a, b):
    #return rbf_kernel(a, b, 0.5)
    return np.dot(a, b)

def normalizeImage(im):
    imd = np.double(im) / 255
    imd = imd - np.amin(imd)
    imd = imd / np.amax(imd)
    imd = np.uint8(imd*255)
    return imd


def grouper(n, iterable, fillvalue=None):
    args = [iter(iterable)] * n
    return itertools.izip_longest(fillvalue=fillvalue, *args)


def grabImgs(cams):
    imgs = []
    states = []
    success = True

    for c in cams:
        if not c.grab():
            success = False

    for c in cams:
        ret, im = c.retrieve()

        if not ret:
            success = False
            continue

        states.append(im[0][0])
        imgs.append(cv2.flip(im, 1))

    return (success, imgs, states)

def grabLightDarkPair(cams):
    success, f1, s1 = grabImgs(cams)
    if not success:
        return (None, None)

    success, f2, s2 = grabImgs(cams)
    if not success:
        return (None, None)

    if s1[0] & 0x20:
        light, dark = f2, f1
    else:
        light, dark = f1, f2

    return (light, dark)


def makeCalibrationImage(w, h, idx):
    ygap = (h - 2*config.calibPadY)/(config.calibRows-1)
    xgap = (w - 2*config.calibPadX)/(config.calibCols-1)
    im = np.zeros([h, w, 3], dtype=np.uint8)
    coord = None

    for r in range(config.calibRows):
        y = config.calibPadY + r*ygap
        for c in range(config.calibCols):
            x = config.calibPadX + c*xgap
            if r*config.calibCols+c == idx:
                coord = (x, y)
                cv2.circle(im, (x,y), 4, (0, 0, 255), -1)
            else:
                cv2.circle(im, (x,y), 4, (70, 70, 0), -1)
    return (im, coord)
