__author__ = 'Luis Bracamontes'

import numpy as np
import cv2
import face2Dalign
import face3Dfront

if __name__ == '__main__':

    #Dlib detector
    lmarkDetector = "shape_predictor_68_face_landmarks.dat"

    #Image to frontalize
    img = cv2.imread('test3.jpg')

    #First align it
    faceAlign = face2Dalign.Face2DAlign(lmarkDetector)
    img = faceAlign.align(img)

    cv2.imshow('Aligned', img)

    #Frontalize the image
    faceFront = face3Dfront.face3Dfront(lmarkDetector)
    print "Rot angle: {0}".format(faceFront.getRotAngle(img))

    frontal_raw, frontal_sym = faceFront.frontalize(img)

    cv2.imshow('Frontalized', frontal_sym)

    print frontal_sym.shape

    cv2.waitKey(0)


