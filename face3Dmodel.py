__author__ = 'Luis Bracamontes'

import scipy.io as scio
import numpy as np

class ThreeD_Model:
    def __init__(self, path, name):
        self.load_model(path, name)

    def load_model(self, path, name):
        model = scio.loadmat(path)[name]
        self.out_A = np.asmatrix(model['outA'][0, 0], dtype='float32') #3x3 Intrinsic matrix
        self.size_U = model['sizeU'][0, 0][0] #1x2 output size
        self.model_TD = np.asarray(model['threedee'][0,0], dtype='float32') #68x3 Landmarks 3D
        self.indbad = model['indbad'][0, 0]#0x1
        self.ref_U = np.asarray(model['refU'][0,0]) #Shape reference