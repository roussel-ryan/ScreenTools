import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

import logging
import h5py

from . import utils
from . import file_processing as fp
from . import image_processing as ip
from . import plotting as plotting
from . import analysis as analysis
from . import current
from .processing import thresholding


def fit_quad_scan(path,L,Bp,conditions=None,plotting = False):
    '''
    use a quad scan about a waist position to calcuate 
    twiss functions and emittance
    filenames in the form of "QuadScan_<gradient>.h5"

    path         string path to folder with images
    L            drift length from quad center to screen in meters
    Bp           magnetic rigidity
    conditions   functions for trimming out frames
    plotting     plot the quadratic fit
    '''

    #get files and extact quad gradients
    files = utils.get_files(path,'.h5')
    gradients = []
    for ele in files:
        gradients.append(ele.split('QuadScan_')[1].split('.h5')[0])
    sort_ind = np.argsort(np.asfarray(gradients))
    files = files[sort_id]
    gradients = gradients[sort_id]

    #find beam sizes for each file
    sx = []
    sy = []
    
    for ele in files:
        stats = analysis.calculate_moments()
        logging.debug(stats)
    
    

def process_files(path):
    pass
