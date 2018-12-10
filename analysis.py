import matplotlib.pyplot as plt
import h5py
import logging
import numpy as np
from scipy.optimize import curve_fit
from . import gaussfitter2
from . import utils

def get_2D_gauss_fit(filename,image_number=0):
    '''
    get the fit parameters for a single image
    - if the image already has been fitted return params
    - returns the fit params via [meanx,meany,stdx,stdy]
    '''

    fit = utils.get_attr(filename,'gauss_fit',dataset='/{}'.format(image_number))

    if not isinstance(fit,np.ndarray):
        #if fit has not been performed then do the fit
        with h5py.File(filename,'r+') as f:
            dataset = f['/{}'.format(image_number)]
            logging.info('doing gaussfit on file:{},image:{}'.format(filename,image_number))
            dx = f.attrs['dx']
            dy = f.attrs['dy']
            
            guess = [1000,1000,dx/2,dy/2,dx/10,dy/10,0.0]
            
            result = gaussfitter2.gaussfit(dataset[:],params=guess)
            logging.info(result)
            
            dataset.attrs['gauss_fit'] = result
            
        return result
    else:
        return fit

def get_array_moments(array):
    '''
    raw calcuation of moments of a 2D histogram array
    
    Inputs
    ------
    array       2D np.ndarray giving an image (histogram) of a distribution
                Note: used in analysis.calc_moments fucntion
    Outputs
    -------
    moments     list(<x>,<y>,<xy>,sqrt(<x^2> - <x>^2),sqrt(<y^2> - <y>^2))

    '''

    T = np.sum(array)
    n,m = array.shape
    
    y = np.arange(n)
    x = np.arange(m)

    exp_x = np.sum(np.dot(array,x),dtype = np.int64) / T
    exp_x2 = np.sum(np.dot(array,(x - exp_x)**2),dtype = np.int64) / T

    exp_y = np.sum(np.dot(array.T,y),dtype = np.int64) / T
    exp_y2 = np.sum(np.dot(array.T,(y - exp_y)**2),dtype = np.int64) / T

    exp_xy = np.sum(np.dot(x,np.dot(array.T,y)),dtype = np.int64) / T
    if exp_y < 0:
        logging.debug(np.dot(array.T,y))
        logging.debug(exp_y)
        logging.debug(T)
    return (exp_x,exp_y,exp_xy,np.sqrt(exp_x2),np.sqrt(exp_y2))
      
def calc_moments(filename):
    '''
    calculate statistical moments for each frame and for the entire file
    results are set as attributes of each image group and in the file attrs

    Inputs
    ------
    filename    h5 file with image data

    Outputs
    -------
    filename    h5 file with image data and results
    '''
    
    stat_data = []
    with h5py.File(filename) as f:
        for i in range(f.attrs['nframes']-1):
            grp = f['/{}'.format(i)]
            stats = get_array_moments(grp['img'][:])
            grp.attrs['moments'] = stats
            stat_data.append(stats)
     
    ndata = np.asfarray(stat_data).T
    stats = []
    for ele in ndata:
        stats.append([np.mean(ele),np.std(ele)])
    nstats = np.asfarray(stats)
    logging.debug(stats)
    with h5py.File(filename,'r+') as f:
        f.attrs['moments_stats'] = stats

    return filename
        

def get_lineout(filename,image_number=0,axis=0,normalize=False,dataset='/img'):
    '''
    calculate the lineout (projection) of a given axis from file
    
    Inputs
    ------
    filename      h5 file with image data
    image_number  frame number in file
    axis          0 for horizontal axis,1 for vertical axis
    normalize     normalizes output so area = 1, defualt=False
    dataset       dataset for image (either /img or /raw)
    '''

    
    with h5py.File(filename,'r') as f:
        data = f['/{}{}'.format(image_number,dataset)][:]
        lineout = np.sum(data,axis=axis)
    if normalize:
        return lineout / np.sum(lineout)
    else:
        return lineout
