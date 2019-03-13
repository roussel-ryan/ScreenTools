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


def fit_quad_scan(path,L,l,Bp,current_to_gradient_func,constraints = None,plotting = False,overwrite = False,base_filename = ('QUADSCAN_','_img.h5'),axis=0,save_points=False,reverse_polarity=False,bounds=None):
    '''
    use a quad scan about a waist position to calcuate 
    twiss functions and emittance
    filenames in the form of "<base_filename>_<current>.h5"

    path         string path to folder with images
    L            drift length from quad center to screen in meters
    l            quadrupole length
    Bp           magnetic rigidity
    current_to_gradient_func            scaling function to go from quad current -> field gradient
    conditions   functions for trimming out frames
    plotting     plot the quadratic fit
    axis         axis for the fit 0:x 1:y
    '''
    s = get_scan_data(path,Bp,current_to_gradient_func,\
                      constraints=constraints,\
                      overwrite=overwrite,base_filename = base_filename,\
                      axis=axis,save_points=save_points,\
                      reverse_polarity=reverse_polarity)

    return fit(s,L,l,plotting=plotting,bounds=bounds,axis=axis)
    

def get_scan_data(path,Bp,current_to_gradient_func,constraints = None,overwrite=False,base_filename = ('QUADSCAN_','_img.h5'),axis=0,save_points=False,reverse_polarity=False):
    #get files and extact quad gradients
    files = utils.get_files(path,'.h5')
    quad_currents = []
    for ele in files:
        quad_currents.append(ele.split(base_filename[0])[1].split(base_filename[1])[0].replace('_','.'))
    sort_index = np.argsort(np.asfarray(quad_currents))
    logging.debug(sort_index)
    files = np.array(files)[sort_index]
    quad_currents = np.asfarray(quad_currents)[sort_index]
    
    #find beam sizes for each file
    s = []

    #logging.debug(gradients)
    
    for f,curr in zip(files,quad_currents):
        try:
            stats = analysis.get_moments(f,constraints,overwrite)
            logging.debug(stats)
            if axis:
                s.append((curr,*stats[1][3]))
            else:
                s.append((curr,*stats[1][0]))
        except RuntimeError:
            pass

    s = np.asfarray(s).T

    if reverse_polarity:
        p = -1.0
    else:
        p = 1.0
    
    #scale current to kappa
    K = p*current_to_gradient_func(s[0]) / Bp

    result = np.array((K,s[1],s[2]))
    
    n = ['x','y']
    if save_points:
        np.savetxt(path + 'quad_scan_{}.txt'.format(n[axis]),result)
    return result

def fit(s,L,l,bounds=None,plotting=False,axis=0):
    f = FitFunction(L,l,axis)
    
    if bounds==None:
        bounds = ((-np.inf,-np.inf,-np.inf),(np.inf,np.inf,np.inf))
    xopt,xcov = optimize.curve_fit(f.fit,s[0],s[1],sigma=s[2],absolute_sigma=True,bounds=bounds)
    #yopt,ycov = optimize.curve_fit(quadratic_fit,sy[0],sy[1],sigma=sy[2],absolute_sigma=True)

    
    if plotting:
        fig,ax = plt.subplots()
        #fig,ax2 = plt.subplots()
        
        ax.errorbar(s[0],s[1],s[2],fmt='o',capsize=5)
        x = np.linspace(s[0][0],s[0][-1])
        ax.plot(x,f.fit(x,*xopt))
        #ax.errorbar(sy[0],sy[1],sx[2],fmt='o',capsize=5)
        #y = np.linspace(sy[0][0],sy[0][-1])
        #ax.plot(y,quadratic_fit(y,*yopt))
        
        ax.set_xlabel('$\kappa [m^{-2}]$')
        ax.set_ylabel('$\sigma^2 [m^2]$')

    #sigma = get_beam_matrix(xopt,xcov,L,l)
    sigma = np.array(((xopt[0],xopt[1]),(xopt[1],xopt[2])))
    
    logging.info('sigma fit: {}'.format(sigma))
    logging.info('Confidence: {}'.format(xcov))

    return sigma
    #return np.array(((xopt[0],xopt[1]),(xopt[1],xopt[2])))

class FitFunction:
    def __init__(self,L,l,axis):
        self.L = L
        self.l = l

        if axis==0:
            self.m = 1
        else:
            self.m = -1
        
    def fit(self,x,s11,s21,s22):
        return s11*(x*self.L*self.l + self.m*1)**2 + \
            2*self.L*(1+x*self.m*self.l*self.L)*s21 + \
            (self.L**2)*s22


def radiabeam_quad_current_to_gradient(current):
    return 0.94*current

def imp_quad_c2g(current):
    return 0.872*current + 0.062

def peach_quad_c2g(current):
    return 0.947*current

def quadratic_fit(x,a,b,c):
    return a*x**2 + b*x + c

def get_beam_matrix(opt,cov,L,l):
    #calculate the beam matrix elements (along with errors) of beam matrix
    s11 = opt[0] / (L*l)**2
    s21 = (opt[1] - 2*L*l*s11) / (2 * L**2 * l)
    s22 = (opt[2] - s11 - 2*L*s21) / (L**2)

#    errors = np.sqrt(cov.diagonal())
#    es11 = errors[0] / (L*l)**2
#    es21 = errors[1] / (2*l*L**2) + es11 / L
#    es22 = errors[2] / L**2 + 2*es21 / L + es11 / L**2
   
#    logging.info('Error of beam matrix @ quad: {}'.format((es11,es21,es22)))
        
    return np.asfarray(((s11,s21),(s21,s22)))
