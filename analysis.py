import matplotlib.pyplot as plt
import h5py
import logging
import numpy as np
from scipy.optimize import curve_fit
from . import gaussfitter2
from . import utils

from . import image_processing as ip

def get_array_moments(array,image_box):
    '''
    raw calcuation of moments of a 2D histogram array
    
    Inputs
    ------
    array       2D np.ndarray giving an image (histogram) of a distribution
                Note: used in analysis.calc_moments fucntion
    Outputs
    -------
    moments     list((<x>,<y>),((<(x-ux)^2>,<(y-uy)^2>,<(x-ux)(y-uy)>)

    '''

    T = np.sum(array)
    n,m = array.shape

    ext = image_box.get_extent()
    #logging.debug(ext)
    y = np.linspace(ext[2],ext[3],n)
    x = np.linspace(ext[0],ext[1],m)

    
    #calculate mean (m) covarience (s) matrix elements
    m1 = np.sum(np.dot(array,x),dtype = np.int64) / T
    s11 = np.sum(np.dot(array,(x - m1)**2),dtype = np.int64) / T

    m2 = np.sum(np.dot(array.T,y),dtype = np.int64) / T
    s22 = np.sum(np.dot(array.T,(y - m2)**2),dtype = np.int64) / T

    s12 = np.sum(np.dot(np.dot(array.T,y - m2),x - m1),dtype = np.int64) / T

    return (np.array((m1,m2)),np.array(((s11,s12),(s12,s22))))

def calculate_ellipse(array,image_box):
    '''
    calculate the parameters for plotting an ellipse on top of an array
    '''
    moments = get_array_moments(array,image_box)
    
    lambda_, v = np.linalg.eig(moments[1])
    return moments[0],lambda_,v

def center_distribution(array,image_box):
    '''
    find center of array, center image_box s.t. beam is at (0,0)
    '''
    moments = get_array_moments(array,image_box)
    image_box.center = image_box.center - moments[0]
    
    return image_box


def calculate_moments(filename,constraints=None):
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
    
    center_data = []
    moment_data = []

    logging.info('Calculating moments of file {}'.format(filename))
    frames = utils.get_frames(filename,constraints)

    
    with h5py.File(filename) as f:
        for i in frames:
            grp = f['/{}'.format(i)]

            ib = ip.ImageBox()
            ib.size = np.asfarray((f['/'].attrs['dx'],f['/'].attrs['dy']))*f['/'].attrs['pixel_scale']
            
            moments = get_array_moments(grp['img'][:],ib)
            grp.attrs['beam_center'] = moments[0]
            grp.attrs['beam_matrix'] = moments[1]
            center_data.append(moments[0])
            moment_data.append(moments[1].flatten())
            
    center_data = np.asfarray(center_data)
    moment_data = np.asfarray(moment_data)
    stats = []
    logging.debug(center_data)

    center_stats = np.array((np.mean(center_data.T,axis=1),np.std(center_data.T,axis=1))).T
    moment_stats = np.array((np.mean(moment_data.T,axis=1),np.std(moment_data.T,axis=1))).T
    
    with h5py.File(filename,'r+') as f:
        f.attrs['beam_center'] = center_stats
        f.attrs['beam_matrix'] = moment_stats
        
    return (center_stats,moment_stats)
        
def get_center_lineout(filename,image_number = 0,axis = 0,dataset='/img'):
    '''
    get the 1D lineout of a given axis from file at the distribution center
    
    Inputs
    ------
    filename      h5 file with image data
    image_number  frame number in file
    axis          0 for horizontal axis,1 for vertical axis
    dataset       dataset for image (either /img or /raw)
    '''
    with h5py.File(filename) as f:
        data = f['/{}{}'.format(image_number,dataset)][:]
        moments = get_array_moments(data,1,1)

    logging.debug(moments)
    if axis:
        return data.T[int(moments[0][0])]
    else:
        return data[int(moments[0][1])]

def get_projection(filename,image_number=0,axis=0,normalize=False,dataset='/img'):
    '''
    calculate the projection of a given axis from file
    
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

def get_averaged_projection(filename,constraint_functions=None):
    '''
    calculate an averaged projection subject to constraint functions

    Inputs
    ------
    filename                 h5 file with image data
    constraint_functions     functions to call to remove particluar 
                             frames, see utils.get_frames
                             Default:None
    '''
    frames = utils.get_frames(filename,constraint_function)
    
    data = []
    for i in frames:
        data.append(get_projection(filename,image_number=i))
    data = np.asfarray(data)
    avg = np.mean(data,axis=0).T
    return avg
    

    
def get_2D_gauss_fit(filename,image_number=0):
    '''
    DEPRECIATED
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
