import cv,cv2

showContours = True
showEyes = True

calibRows = 5
calibCols = 5
calibPadX = 300
calibPadY = 300

eyePatchSize = 70

pupilThresh1Min = 60
pupilThresh1Max = 255
pupilThresh2Min = 70
pupilThresh2Max = 255
glintThreshMin  = 150
glintThreshMax  = 255

def setPupilThresh1Min(arg):
    global pupilThresh1Min
    pupilThresh1Min = arg
def setPupilThresh1Max(arg):
    global pupilThresh1Max
    pupilThresh1Max = arg
def setPupilThresh2Min(arg):
    global pupilThresh2Min
    pupilThresh2Min = arg
def setPupilThresh2Max(arg):
    global pupilThresh2Max
    pupilThresh2Max = arg
def setGlintThreshMin(arg):
    global glintThreshMin
    glintThreshMin = arg
def setGlintThreshMax(arg):
    global glintThreshMax
    glintThreshMax = arg
