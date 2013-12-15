import cv,cv2
import numpy as np
import itertools
import sys
import math

def rbf_kernel(a, b, sigma):
    print np.exp(-(1/(2*sigma*sigma))*(LA.norm(a - b)*LA.norm(a - b)))
    return np.exp(-(1/(2*sigma*sigma))*(LA.norm(a - b)*LA.norm(a - b)))

def kernel(a, b):
    #return rbf_kernel(a, b, 0.5)
    return np.dot(a, b)

def contourFilterFunc(c):
    hull = cv2.convexHull(c)
    arcLen = cv2.arcLength(c, True)
    hullArcLen = cv2.arcLength(hull, True)
    #area = cv2.contourArea(c)
    #hullArea = cv2.contourArea(hull)
    #return area == 0 or hullArea/area <= 1.00000000001
    return hullArcLen == 0 or arcLen/hullArcLen <= 1.2

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
        ret, f = c.read()

        if not ret:
            success = False
            break

        states.append(f[0][0])
        f = cv2.flip(f, 1)

        imgs.append(f)

    return (success, imgs, states)

def getEye(im, contour, sz):
    moments = cv2.moments(contour)
    if moments['m00'] > 0:
        x, y = moments['m10']/moments['m00'], moments['m01']/moments['m00']
        half = int(sz/2)
        roi = im[int(y)-half:int(y)+half, int(x)-half:int(x)+half]
        if roi.size == sz*sz:
            return (x,y), normalizeImage(roi.copy())

    return None, None 

def getCircle(eye, thresh):
    eyed = normalizeImage(eye)

    diffThresh = cv2.inRange(eyed, thresh, 255)
    diffCont = diffThresh.copy()
    contours, hier = cv2.findContours(diffCont, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
     
    def filterFunc(c):
        hull = cv2.convexHull(c)
        arcLen = cv2.arcLength(c, True)
        hullArcLen = cv2.arcLength(hull, True)
        area = cv2.contourArea(c)
        hullArea = cv2.contourArea(hull)
        return (area == 0       or hullArea/area <= 1.4)
               #(hullArcLen == 0 or arcLen/hullArcLen <= 1.05)

    contours = filter(filterFunc, contours)
    maxContour = None;
    maxContourArea = 0;
    for c in contours:
        area = cv2.contourArea(c)
        if area > maxContourArea:
            maxContourArea = area
            maxContour = c

    if maxContour != None:
        moments = cv2.moments(maxContour)
        if moments['m00'] > 0:
            return (moments['m10']/moments['m00'], moments['m01']/moments['m00']), contours

    return (None, contours)

def getGlint(eye, pupilCent, thresh, eyeBoxSize):
    eyed = normalizeImage(eye)

    diffThresh = cv2.inRange(eyed, thresh, 255)
    diffCont = diffThresh.copy()
    contours, hier = cv2.findContours(diffCont, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
     
    def filterFunc(c):
        hull = cv2.convexHull(c)
        arcLen = cv2.arcLength(c, True)
        hullArcLen = cv2.arcLength(hull, True)
        area = cv2.contourArea(c)
        hullArea = cv2.contourArea(hull)
        return (area == 0       or hullArea/area <= 1.5)
               #(hullArcLen == 0 or arcLen/hullArcLen <= 1.05)

    #contours = filter(filterFunc, contours)
    minDistance = sys.maxint;
    glint = None
    for c in contours:
        moments = cv2.moments(c)
        if moments['m00'] > 0:
            x, y = moments['m10']/moments['m00'], moments['m01']/moments['m00']
            dist = math.sqrt(math.pow(x - (pupilCent[0] - int(eyeBoxSize/4)), 2) + math.pow(y - (pupilCent[1] - int(eyeBoxSize/4)), 2))
            if dist < minDistance:
                minDistance = dist
                glint = (x, y)

    if glint != None:
        return glint, contours

    return (None, contours)

def makeCalibrationImage(w, h, rows, cols, idx):
    pad = 250
    ygap = (h - 2*pad)/(rows-1)
    xgap = (w - 2*pad)/(cols-1)
    im = np.zeros([h, w, 3], dtype=np.uint8)
    coord = None

    for r in range(rows):
        y = pad + r*ygap
        for c in range(cols):
            x = pad + c*xgap
            if r*cols+c == idx:
                coord = (x, y)
                cv2.circle(im, (x,y), 4, (0, 0, 255), -1)
            else:
                cv2.circle(im, (x,y), 4, (70, 70, 0), -1)
    return (im, coord)
