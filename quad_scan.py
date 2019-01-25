import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
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


def fit_quad_scan(path,L,l,Bp,current_to_gradient_func,conditions = None,plotting = False,overwrite = False):
    '''
    use a quad scan about a waist position to calcuate 
    twiss functions and emittance
    filenames in the form of "QuadScan_<gradient>.h5"

    path         string path to folder with images
    L            drift length from quad center to screen in meters
    l            quadrupole length
    Bp           magnetic rigidity
    current_to_gradient_func            scaling function to go from quad current -> field gradient
    conditions   functions for trimming out frames
    plotting     plot the quadratic fit
    '''

    #get files and extact quad gradients
    files = utils.get_files(path,'.h5')
    quad_currents = []
    for ele in files:
        quad_currents.append(ele.split('QUADSCAN_')[1].split('_img.h5')[0].replace('_','.'))
    sort_index = np.argsort(np.asfarray(quad_currents))
    logging.debug(sort_index)
    files = np.array(files)[sort_index]
    quad_currents = np.asfarray(quad_currents)[sort_index]
    
    #find beam sizes for each file
    sx = []
    sy = []

    #logging.debug(gradients)
    
    for f,curr in zip(files,quad_currents):
        stats = analysis.get_moments(f,conditions,overwrite)
        logging.debug(stats)
        if curr > 0.0:
            
            sx.append((curr,*stats[1][0]))
        else:
            sy.append((curr,*stats[1][3]))

    sx = np.asfarray(sx).T
    sy = np.asfarray(sy).T


    #scale current to kappa
    sx[0] = current_to_gradient_func(sx[0]) / Bp
    sy[0] = current_to_gradient_func(sy[0]) / Bp

    #scale current to (1 + LlK) for fitting
    #fit_x = 1 + sx[0]*L*l
    #fit_y = 1 + sy[0]*L*l

    #do fitting
    xopt,xcov = optimize.curve_fit(quadratic_fit,sx[0],sx[1],sigma=sx[2],absolute_sigma=True)
    yopt,ycov = optimize.curve_fit(quadratic_fit,sy[0],sy[1],sigma=sy[2],absolute_sigma=True)

    
    if plotting:
        fig,ax = plt.subplots()
        #fig,ax2 = plt.subplots()
        
        ax.errorbar(sx[0],sx[1],sx[2],fmt='o',capsize=5)
        x = np.linspace(sx[0][0],sx[0][-1])
        ax.plot(x,quadratic_fit(x,*xopt))
        ax.errorbar(sy[0],sy[1],sx[2],fmt='o',capsize=5)
        y = np.linspace(sy[0][0],sy[0][-1])
        ax.plot(y,quadratic_fit(y,*yopt))
        
        ax.set_xlabel('$\kappa [m^{-2}]$')
        ax.set_ylabel('$\sigma^2 [m^2]$')

    xmatrix = get_beam_matrix(*xopt,L,l)
    ymatrix = get_beam_matrix(*yopt,L,l)
    logging.debug(xmatrix)
    logging.debug(ymatrix)
    logging.debug(np.sqrt(np.linalg.det(xmatrix)))
    logging.debug(np.sqrt(np.linalg.det(ymatrix)))
    
    return (xmatrix,ymatrix)


def radiabeam_quad_current_to_gradient(current):
    return 0.94*current

def quadratic_fit(x,a,b,c):
    return a*x**2 + b*x + c

def get_beam_matrix(A,B,C,L,l):
    s11 = A / (L**2 * l**2)
    s21 = (B - 2*L*l*s11) / (2 * L**2 * l)
    s22 = (C - s11 - 2*L*s21) / L **2
    return np.asfarray(((s11,s21),(s21,s22)))
