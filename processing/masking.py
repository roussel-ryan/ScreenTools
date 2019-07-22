import numpy as np
import numpy.ma as ma
import logging
import h5py
import matplotlib.pyplot as plt

from scipy import ndimage

from .. import utils

def get_filled_rowcol(array):
    '''
    remove columns and rows containing only zeros
    '''

    #do for one axis
    empty_test = np.sum(array,axis=1)
    #rows = np.argwhere(empty_test).flatten()].T

    # and the same for the other
    empty_test = np.sum(array,axis=1)
    return array[np.argwhere(empty_test).flatten()].T

def get_mask(array,center,radius):
    '''
    set all elements outside of circular screen to zero
    '''
    shape = array.shape
    r = 0.875*radius
    mask = np.ones_like(array)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            if np.sqrt((j - center[0])**2 + (i - center[1])**2) < r:
                mask[i][j] = 0
        
    return mask

def remove_empty_space(narray):
    #remove all cols and rows where every value is masked
    narray = narray[:, ~np.all(narray.mask, axis=0)].T
    narray = narray[:, ~np.all(narray.mask, axis=0)].T
    return narray    

def mask(h5file,frame_number=-1,overwrite=False):
    '''
    masks all images in <h5file>
    '''
    #logging.debug('masking file {}'.format(h5file))
    frames = utils.check_frame_number(h5file,frame_number)

    with h5py.File(h5file,'r+') as f:
        #check which frames are not masked
        unmasked = []
        for ele in frames:
            try:
                m = f['/{}'.format(ele)].attrs['masked']
                if overwrite:
                    m = 0
            except KeyError:
                m = 0

            if not m:
                unmasked.append(ele)

        #get masking array
        frames = f.attrs['nframes']
        center = f.attrs['screen_center']
        radius = f.attrs['screen_radius']

        try:
            dataset = f['/{}/img'.format(unmasked[0])]
        except IndexError:
            logging.warning('No unmasked frames in {}'.format(h5file))
            return None
        
        narray = np.array(dataset[:])
        mask = get_mask(narray,center,radius)

        for i in unmasked:
            logging.debug('Doing masking of file: {}, frame: {}'.format(h5file,i))
            grp = f['/{}'.format(i)]
            dataset = grp['img'][:]
            narray = np.array(dataset)
            narray = ma.array(narray,mask = mask)
            narray = remove_empty_space(narray)
            narray = ma.filled(narray,0)
                    
            del grp['img']
            
            grp.create_dataset('img',data=narray)
            grp.attrs['masked'] = 1
            
    return None
