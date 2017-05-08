__author__ = 'Luis Bracamontes'

import numpy as np
import dlib
import cv2
import math

class Face2DAlign:
    """Carry out 2D aligment"""
    __MEAN_FACE_FILE = './mean_vals2.txt'
    __INDEX_MEAN_FACE = [0, 7, 14, 18, 15, 21, 24, 27, 29, 32, 34, 48, 54, 67, 2, 12]
    __INDEX_DLIB_FACE = [0, 8, 16, 22, 26, 17, 21, 36, 39, 45, 42, 48, 54, 30, 2, 14]

    __WIDTH = 266
    __HEIGHT = 294

    __LMARKSRC1 = 0
    __LMARKSRC2 = 4

    def __init__(self, lmarkPredictor):
        """
        Instantiate an 'AlignDlib' object.

        :param lmarkPredictor: The path to dlib's
        :type lmarkPredictor: str
        """
        assert lmarkPredictor is not None

        self.faceLocator = dlib.get_frontal_face_detector()
        self.lmarkLocator = dlib.shape_predictor(lmarkPredictor)

    def ladnmarksToNpArray(self, lmarks):
        """
        Returns face landmarks as a numpy array
        """

        np_lmarks = np.zeros((68, 2), dtype='int')

        #Go over all dlib lanmarks
        for i in range(0, 68):
            np_lmarks[i] = (lmarks.part(i).x, lmarks.part(i).y)

        return np_lmarks

    def getLargestFace(self, img):
        """
        Find Largest face in an image.

        :param img: RGB image to process. Shape: (height, width, 3)
        :type img: numpy.ndarray
        :return: Face bounding box in an image.
        :rtype: dlib.rectangles
        """
        assert img is not None

        try:
            faces = self.faceLocator(img, 1)
            if len(faces) > 0:
                return max(faces, key=lambda rect: rect.width() * rect.height())
            else:
                raise ValueError("No faces detected!")
        except Exception as e:
            print "Warning: {}".format(e)
            # In rare cases, exceptions are thrown.
            return []

    def getLmarks(self, img, face, All=False):
        """
        Find landmarks in an a face

        :param img: RGB image to process. Shape: (height, width, 3)
        :type img: numpy array
        :param face: Bounding box of a face
        :type: dlib.rectangle
        :rtype: list of landmarks
        """

        assert img is not None
        assert face is not None

        try:
            lmarks = self.lmarkLocator(img, face)
        except TypeError:
            print "There is a problem with the lanmarks!"
            return None
        lmarks = self.ladnmarksToNpArray(lmarks).tolist()

        lmarksSrc = []
        if not All:
            for n, (x, y) in enumerate(lmarks):
                if n in Face2DAlign.__INDEX_DLIB_FACE:
                    lmarksSrc.append((x, y))
        else:
            lmarksSrc = lmarks

        return lmarksSrc

    def getMeanLmarks(self):
        """
        Returns mean faces landmark location
        :rtype: numpy array
        """

        fil = open(Face2DAlign.__MEAN_FACE_FILE)
        lmarkDes = []
        for n, line in enumerate(fil):
            if n in Face2DAlign.__INDEX_MEAN_FACE:
                lmarkDes.append((int(line.rsplit(' ')[0]), int(line.rsplit(' ')[1])))

        return lmarkDes

    def __matA(self, lmarkSrc):
        """
        Computes Tranformation matrix
        :param lmarkSrc: Input vectors
        :type lmarkSrc: list
        :rtype: numpy ndarray
        """

        elem = 2*len(lmarkSrc)

        A = np.zeros((elem, 4))

        for i, pnt in enumerate(lmarkSrc):
            a = np.array([0, 0, 0, 0], dtype=np.float32)
            a[0] = pnt[0]
            a[1] = -1*pnt[1]
            a[2] = 1
            a[3] = 0
            A[2*i] = a
            a[0] = pnt[1]
            a[1] = pnt[0]
            a[2] = 0
            a[3] = 1
            A[2*i+1] = a

        return A

    def __vecB(self, lmarkDes):
        """
        Computes vector for desired value
        :param lmarkSrc: Input vectors
        :type lmarkSrc: list
        :rtype: numpy array
        """
        lmarkDes = np.array(lmarkDes)
        return lmarkDes.flatten()

    def __solve(self, tsnMat, vecB):
        """
        Solves LSQ to find optimum values
        :param tsnMat: transformation matrix
        :type vetsnMat: numpy ndarray
        :param vecB: transformation matrix
        :type vecB: numpy ndarray
        :rtype: numpy array
        """
        A_d = np.linalg.inv(np.dot(tsnMat.T, tsnMat))
        return np.dot(np.dot(A_d, tsnMat.T), vecB)

    def similarity2DTransform(self, lmarkSrc, lmarksDes):
        """
        Compute similarity transformation
        :param lmarkSrc: Input points
        :type lmarkSrc: list
        :param lmarkDes: Destination points or desire output
        :type lmarkDes: list
        :rtype 2x3 transform matrix
        """
        assert lmarkSrc is not None and len(lmarkSrc) > 0
        assert lmarksDes is not None and len(lmarksDes) > 0

        H = self.__matA(lmarkSrc)
        b = self.__vecB(lmarksDes)
        x = self.__solve(H, b)
        s = 1.0

        M = np.zeros((2, 3))
        M[0] = np.array([s*x[0], -s*x[1], x[2]])
        M[1] = np.array([s*x[1], s*x[0], x[3]])

        return M

    def rigid2PointTransform(self, lmarkSrc, lmarksDes):
        """
        Compute rigid transformation from 2 points
        :param lmarkSrc: Input points
        :type lmarkSrc: list
        :param lmarkDes: Destination points or desire output
        :type lmarkDes: list
        :rtype 2x3 transform matrix
        """

        s60 = math.sin(60*math.pi/180)
        c60 = math.cos(60*math.pi/180)

        inPts = np.copy(lmarkSrc).tolist()
        outPts = np.copy(lmarksDes).tolist()

        xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0]
        yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1]

        inPts.append([np.int(xin), np.int(yin)])

        xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0]
        yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1]

        outPts.append([np.int(xout), np.int(yout)])

        tform = cv2.estimateRigidTransform(np.array([inPts]), np.array([outPts]), False)

        return tform

    def align(self, img):
        """
        Align Image.

        :param img: Image containg a face
        :type numpy array: str
        :rtype numpy array of aligned image
        """

        assert img is not None

        face = self.getLargestFace(img)
        lmarkSrc = self.getLmarks(img, face)
        lmarkDes = self.getMeanLmarks()

        tform = self.similarity2DTransform(lmarkSrc, lmarkDes)

        img_aligned = cv2.warpAffine(img, tform, (Face2DAlign.__WIDTH, Face2DAlign.__HEIGHT))

        face = self.getLargestFace(img_aligned)
        lmarkSrc = self.getLmarks(img_aligned, face)

        lmarkSrc = np.float32([lmarkSrc[Face2DAlign.__LMARKSRC1], lmarkSrc[Face2DAlign.__LMARKSRC2]])
        lmarkDes = np.float32([[np.int(0.05 * Face2DAlign.__WIDTH ), np.int(0.3 * Face2DAlign.__HEIGHT)],
                    [np.int(0.95 * Face2DAlign.__WIDTH ), np.int(0.3 * Face2DAlign.__HEIGHT)]])

        tform = self.rigid2PointTransform(lmarkSrc, lmarkDes)

        return cv2.warpAffine(img_aligned, tform, (Face2DAlign.__WIDTH, Face2DAlign.__HEIGHT))
