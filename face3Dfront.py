__author__ = 'Luis Bracamontes'

import numpy as np
import cv2
import dlib
import scipy.io as scio
import camera_calibration as calib
import scipy.io as io
import face3Dmodel 

class face3Dfront:

    __MEAN_3D_FACE_FILE = './frontalization_models/model3Ddlib.mat'
    __3D_FACE_MODEL = 'model_dlib'
    __EYEMASK = './frontalization_models/eyemask.mat'
    __ACC_CONST = 800

    def __init__(self, lmarkPredictor):
        """
        Instantiate an 'AlignDlib' object.
        :param lmarkPredictor: The path to dlib's
        :type lmarkPredictor: str
        """
        assert lmarkPredictor is not None

        self.faceLocator = dlib.get_frontal_face_detector()
        self.lmarkLocator = dlib.shape_predictor(lmarkPredictor)

    def __loadEyemask(self):
        """
        Return eyemask of the face.
        :return: Eyemask for the face
        :rtype: numpy array
        """
        return np.asarray(io.loadmat(face3Dfront.__EYEMASK)['eyemask'])

    def cropFrontalFace(self, img):
        """
        Return cropped face based on landmarks location.
        :param img: symetric frontalized image
        :type img: numpy array
        :return: Cropped image.
        :rtype: numpy array
        """
        assert img is not None

        lmarks = self.getLandmarks(img)

        top = int(lmarks[36][1] - 36)
        left = int(lmarks[36][0] - 13)
        width = int(lmarks[45][0] + 35)
        height = int(lmarks[57][1] + 13)

        img =  cv2.resize(img[top:width, left:height], (120, 145))

        return img

    def __interFrontalize(self, img, proj_matrix, ref_U, eyemask):
        img = img.astype('float32')
        print "query image shape:", img.shape

        bgind = np.sum(np.abs(ref_U), 2) == 0
        # count the number of times each pixel in the query is accessed
        threedee = np.reshape(ref_U, (-1, 3), order='F').transpose()
        temp_proj = proj_matrix * np.vstack((threedee, np.ones((1, threedee.shape[1]))))
        temp_proj2 = np.divide(temp_proj[0:2, :], np.tile(temp_proj[2, :], (2,1)))

        bad = np.logical_or(temp_proj2.min(axis=0) < 1, temp_proj2[1, :] > img.shape[0])
        bad = np.logical_or(bad, temp_proj2[0, :] > img.shape[1])
        bad = np.logical_or(bad, bgind.reshape((-1), order='F'))
        bad = np.asarray(bad).reshape((-1), order='F')

        nonbadind = np.nonzero(bad == 0)[0]
        temp_proj2 = temp_proj2[:, nonbadind]
        # because python arrays are zero indexed
        temp_proj2 -= 1
        ind = np.ravel_multi_index((np.asarray(temp_proj2[1, :].round(), dtype='int64'), np.asarray(temp_proj2[0, :].round(),
                                dtype='int64')), dims=img.shape[:-1], order='F')
        synth_frontal_acc = np.zeros(ref_U.shape[:-1])
        ind_frontal = np.arange(0, ref_U.shape[0]*ref_U.shape[1])
        ind_frontal = ind_frontal[nonbadind]
        c, ic = np.unique(ind, return_inverse=True)
        bin_edges = np.r_[-np.Inf, 0.5 * (c[:-1] + c[1:]), np.Inf]
        count, bin_edges = np.histogram(ind, bin_edges)
        synth_frontal_acc = synth_frontal_acc.reshape(-1, order='F')
        synth_frontal_acc[ind_frontal] = count[ic]
        synth_frontal_acc = synth_frontal_acc.reshape((320, 320), order='F')
        synth_frontal_acc[bgind] = 0
        synth_frontal_acc = cv2.GaussianBlur(synth_frontal_acc, (15, 15), 30., borderType=cv2.BORDER_REPLICATE)
        frontal_raw = np.zeros((102400, 3))
        frontal_raw[ind_frontal, :] = cv2.remap(img, temp_proj2[0, :].astype('float32'), temp_proj2[1, :].astype('float32'), cv2.INTER_CUBIC)
        frontal_raw = frontal_raw.reshape((320, 320, 3), order='F')

        # which side has more occlusions?
        midcolumn = np.round(ref_U.shape[1]/2)
        sumaccs = synth_frontal_acc.sum(axis=0)
        sum_left = sumaccs[0:midcolumn].sum()
        sum_right = sumaccs[midcolumn+1:].sum()
        sum_diff = sum_left - sum_right

        if np.abs(sum_diff) > face3Dfront.__ACC_CONST: # one side is ocluded
            ones = np.ones((ref_U.shape[0], midcolumn))
            zeros = np.zeros((ref_U.shape[0], midcolumn))
            if sum_diff > face3Dfront.__ACC_CONST: # left side of face has more occlusions
                weights = np.hstack((zeros, ones))
            else: # right side of face has more occlusions
                weights = np.hstack((ones, zeros))
            weights = cv2.GaussianBlur(weights, (33, 33), 60.5, borderType=cv2.BORDER_REPLICATE)

            # apply soft symmetry to use whatever parts are visible in ocluded side
            synth_frontal_acc /= synth_frontal_acc.max()
            weight_take_from_org = 1. / np.exp(0.5+synth_frontal_acc)
            weight_take_from_sym = 1 - weight_take_from_org

            weight_take_from_org = np.multiply(weight_take_from_org, np.fliplr(weights))
            weight_take_from_sym = np.multiply(weight_take_from_sym, np.fliplr(weights))

            weight_take_from_org = np.tile(weight_take_from_org.reshape(320, 320, 1), (1, 1, 3))
            weight_take_from_sym = np.tile(weight_take_from_sym.reshape(320, 320, 1), (1, 1, 3))
            weights = np.tile(weights.reshape(320, 320, 1), (1, 1, 3))

            denominator = weights + weight_take_from_org + weight_take_from_sym
            frontal_sym = np.multiply(frontal_raw, weights) + np.multiply(frontal_raw, weight_take_from_org) + np.multiply(np.fliplr(frontal_raw), weight_take_from_sym)
            frontal_sym = np.divide(frontal_sym, denominator)

            # exclude eyes from symmetry
            frontal_sym = np.multiply(frontal_sym, 1-eyemask) + np.multiply(frontal_raw, eyemask)
        else: # both sides are occluded pretty much to the same extent -- do not use symmetry
            frontal_sym = frontal_raw

        frontal_raw[frontal_raw > 255] = 255
        frontal_raw[frontal_raw < 0] = 0
        frontal_raw = frontal_raw.astype('uint8')
        frontal_sym[frontal_sym > 255] = 255
        frontal_sym[frontal_sym < 0] = 0
        frontal_sym = frontal_sym.astype('uint8')

        frontal_sym = self.cropFrontalFace(frontal_sym)

        return frontal_raw, frontal_sym

    def ladnmarksToNp(self, shape, dtype='int'):
        """Reads detected face landmarks and returns 
        them as a numpy array 
        """
        coords = np.zeros((68,2), dtype=dtype)

        #Go over all dlib lanmarks
        for i in range(0,68):
           coords[i] = (shape.part(i).x, shape.part(i).y)

        return coords

    def getLandmarks(self, img):
        """
        Find landmarks in biggest face in an image.

        :param img: RGB image to process. Shape: (height, width, 3)
        :type img: numpy.ndarray
        :return: Landmark location of bigger face.
        :rtype: list.
        """
        assert img is not None

        try:
            faces = self.faceLocator(img, 1)
            if len(faces) > 0:
                bigFace =  max(faces, key=lambda rect: rect.width() * rect.height())
            else:
                raise ValueError("No faces detected!")
        except Exception as e:
            print "Warning: {}".format(e)
            # In rare cases, exceptions are thrown.
            return []

        lmarks = self.lmarkLocator(img, bigFace)
        lmarks = self.ladnmarksToNp(lmarks)

        lmarks = np.asarray(lmarks, dtype='float32')

        return lmarks

    def getRotAngle(self, img):
        """
        Returns rotation angle for yaw thta is necessary to frontalize.
        :param img: Image to be frontalize.
        :type img: numpy array
        :return: Rotation angle degrees.
        :rtype: float
        """
        assert img is not None

        face3DModel = face3Dmodel.ThreeD_Model(face3Dfront.__MEAN_3D_FACE_FILE, face3Dfront.__3D_FACE_MODEL)

        lmarks = self.getLandmarks(img)

        # perform camera calibration according to the first face detected
        proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(face3DModel, lmarks)

        return calib.get_yaw(rmat)


    def frontalize(self, img):
        """
        Carries out frontalization of biggest face in an image.
        :param img: Image with faces.
        :type img: numpy array
        :return: Frontalized image
        """

        assert img is not None

        face3DModel = face3Dmodel.ThreeD_Model(face3Dfront.__MEAN_3D_FACE_FILE, face3Dfront.__3D_FACE_MODEL)

        lmarks = self.getLandmarks(img)

        # perform camera calibration according to the first face detected
        proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(face3DModel, lmarks)

        eyemask = self.__loadEyemask()

        img = img.astype('float32')

        return self.__interFrontalize(img, proj_matrix, face3DModel.ref_U, eyemask)
