import numpy as np
import numpy.ma as ma
import logging
import h5py
import matplotlib.pyplot as plt

from scipy import ndimage

from .processing import thresholding
from .processing import masking
from . import utils

def mask_file(h5file):
    '''
    masks all images in <h5file>
    '''
    return masking.mask_file(h5file)
    

def threshold_file(h5file,level = 0, manual = False):
    logging.info('thresholding {}'.format(h5file))
    if manual:
        m = thresholding.ManualThresholdSelector(h5file)
    else:
        thresholding.set_threshold(h5file,level,manual)

    thresholding.apply_threshold(h5file)
    return h5file
                

def aperature_array(array,center,radius,plotting = False):
    data = []
    r = np.linspace(0,radius)
    
    
    for radius in r:
        mask = get_mask(array,center,radius)
        data.append([radius,np.sum(ma.array(array,mask = mask))])
    ndata = np.asfarray(data).T

    if plotting:
        fig,ax = plt.subplots()
        ax.plot(ndata[0],ndata[1])
    
def filter_array(array,sigma=1,size=4):
    ''' apply a median filter and a gaussian filter to image'''
    ndata = ndimage.median_filter(array,size)
    return ndimage.gaussian_filter(ndata,sigma)
    
def filter_file(h5file,filter_high_res = False):
    '''filtering'''
    if not utils.get_attr(h5file,'filtered'):
        with h5py.File(h5file,'r+') as f:
            logging.info('applying gaussian filter to file {}'.format(h5file))

            for i in range(f.attrs['nframes']-1):
                logging.debug('filtering frame {}'.format(i))
                dataset = f['/{}/img'.format(i)]
                narray = filter_array(dataset[:])
                dataset[...] = narray
            f.attrs['filtered'] = 1
    
def reset_image(h5file,image_number=0):
    logging.debug('resetting image {}'.format(image_number,h5file))
    with h5py.File(h5file,'r+') as f:
        del f['/{}/img'.format(image_number)]
        raw = f['/{}/raw'.format(image_number)][:]

        f.create_dataset('/{}/img'.format(image_number),data=raw)
        
def reset_file(h5file):
    logging.info('resetting file {}'.format(h5file))
    for i in range(utils.get_attr(h5file,'nframes')-1):
        reset_image(h5file,i)
    with h5py.File(h5file,'r+') as f:
        names = ['filtered','background_removed','masked','global_threshold']
        for name in names:
            try:
                del f.attrs[name]
            except KeyError:
                pass
