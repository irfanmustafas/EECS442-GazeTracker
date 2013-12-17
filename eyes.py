import cv,cv2
import math

import config
from helper import *
import svm

def contourFilterFunc(c):
    hull = cv2.convexHull(c)
    arcLen = cv2.arcLength(c, True)
    hullArcLen = cv2.arcLength(hull, True)
    #area = cv2.contourArea(c)
    #hullArea = cv2.contourArea(hull)
    #return area == 0 or hullArea/area <= 1.00000000001
    return hullArcLen == 0 or arcLen/hullArcLen <= 1.3

def findBlobs(light, dark):
    pupilContours = []
    displayImgs = []
    for s in zip(light, dark):
        diff = cv2.absdiff(s[0], s[1])
  
        diffThresh = cv2.inRange(diff, config.pupilThresh1Min, config.pupilThresh1Max)
        #diffThresh = cv2.adaptiveThreshold(diffStereo, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, maxThreshold)
        diffThresh = cv2.dilate(diffThresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)))

        contours, hier = cv2.findContours(diffThresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = filter(contourFilterFunc, contours)
        pupilContours.append(contours)

        if config.showContours:
            imCont = np.zeros((diffThresh.shape[0], diffThresh.shape[1], 3), dtype=np.uint8)
            cv2.drawContours(imCont, contours, -1, (255, 255, 255))
            displayImgs.append(imCont)

    if config.showContours:
        cv2.imshow('contours', np.concatenate(displayImgs, axis=1))

    return pupilContours

def findEyes(light, dark, contours, imboth):
    pupilGlintVectors = []

    for cam in range(len(light)):
        PGVec = []

        mainPupil = None
        mainGlint = None
        minDist = sys.maxint
        mainBox = None

        for c in contours[cam]:
            (pos, eye) = getEyePatch(light[cam], c)

            if eye is not None:
                x, y = int(pos[0]), int(pos[1])
                half    = int(config.eyePatchSize / 2)
                quarter = int(config.eyePatchSize / 4)

                label = np.dot(svm.w, svm.scale * (np.reshape(eye, config.eyePatchSize*config.eyePatchSize, 'F') + svm.shift)) + svm.bias
                if label <= 0:
                    pupil = getPupil(eye)

                    if config.showEyes:
                        cv2.rectangle(imboth, (x-half,y-half), (x+half,y+half), (255, 255, 255), 2)

                    if pupil is not None:
                        darkEye = dark[cam][y-half+int(pupil[1])-quarter:y-half+int(pupil[1])+quarter, x-half+int(pupil[0])-quarter:x-half+int(pupil[0])+quarter]
                        if darkEye.size != math.pow(2*quarter,2):
                            continue

                        darkEye = normalizeImage(darkEye)
                        glint = getGlint(darkEye)

                        #imGlint = np.zeros([2*quarter, 2*quarter])
                        #cv2.drawContours(imGlint, gCont, -1, (255, 255, 255))
                        #eyeContours = np.concatenate([eyeContours, imGlint], axis=1)

                        dist = math.sqrt(math.pow(360 - (x-half+int(pupil[0])), 2) + math.pow(240 - (y-half+int(pupil[1])), 2))
                        if dist < minDist and glint is not None:
                            minDist = dist
                            mainPupil = (x-half+pupil[0], y-half+pupil[1])
                            mainGlint = (mainPupil[0]-quarter+glint[0], mainPupil[1]-quarter+glint[1])
                            mainBox = (x-half, y-half), (x+half, y+half)
                            #cv2.circle(imboth, (x-int(config.eyePatchSize/4)+int(glint[0]), y-int(config.eyePatchSize/4)+int(glint[1])), 3, (0, 0, 255))
        
        if mainPupil is not None and mainGlint is not None:
            PGVec.append((mainPupil[0] - mainGlint[0], mainPupil[1] - mainGlint[1]))
            if config.showEyes:
                cv2.circle(imboth, (int(mainPupil[0]), int(mainPupil[1])), 7, (0, 255, 0))
                cv2.circle(imboth, (int(mainGlint[0]), int(mainGlint[1])), 3, (0, 0, 255))
                cv2.rectangle(imboth, mainBox[0], mainBox[1], (0, 255, 0), 2)

        pupilGlintVectors.append(PGVec)

    return pupilGlintVectors


def getEyePatch(im, contour):
    moments = cv2.moments(contour)
    if moments['m00'] > 0:
        x, y = moments['m10']/moments['m00'], moments['m01']/moments['m00']
        half = int(config.eyePatchSize / 2)
        roi = im[int(y)-half:int(y)+half, int(x)-half:int(x)+half]
        if roi.size == config.eyePatchSize * config.eyePatchSize:
            return (x,y), normalizeImage(roi.copy())

    return None, None 

def getPupil(eye):
    edges = cv2.Canny(eye, 200, 400)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    contours, hier = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    maxArea = 0
    pupil = None
    cont = None
    for c in contours:
        moments = cv2.moments(c)
        if moments['m00'] > maxArea:
            maxArea = moments['m00']
            pupil = (moments['m10']/moments['m00'], moments['m01']/moments['m00'])
            cont = c

    if pupil is not None and cont.shape[0] >= 5:
        ell = cv2.fitEllipse(cont)
        return ell[0]
    else:
        return None


def getGlint(eye):
    edges = cv2.Canny(eye, 350, 600)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    contours, hier = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    maxArea = 0
    glint = None
    for c in contours:
        moments = cv2.moments(c)
        if moments['m00'] > maxArea:
            maxArea = moments['m00']
            glint = (moments['m10']/moments['m00'], moments['m01']/moments['m00'])

    if glint is not None:
        cv2.circle(eye, (int(glint[0]), int(glint[1])), 4, (255,255,255), 1)
    cv2.imshow('EYE', eye)

    return glint

    """
    diffThresh = cv2.inRange(eye, thresh, 255)
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
            dist = math.sqrt(math.pow(x - (pupilCent[0] - int(config.eyePatchSize/4)), 2) + math.pow(y - (pupilCent[1] - int(config.eyePatchSize/4)), 2))
            if dist < minDistance:
                minDistance = dist
                glint = (x, y)

    if glint != None:
        return glint, contours

    return (None, contours)
    """
