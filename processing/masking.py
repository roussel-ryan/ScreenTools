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

def mask_frame(h5file,frame_number):
    with h5py.File(h5file,'r+') as f:
        center = f.attrs['screen_center']
        radius = f.attrs['screen_radius']
            
        dataset = f.get('/{}/img'.format(frame_number))
        narray = np.array(dataset)
        mask = get_mask(narray,center,radius)

        narray = ma.array(narray,mask = mask)
        narray = remove_empty_space(narray)
               
        #fill masked values with zeros
        narray = ma.filled(narray,0)

        del f['/{}/img'.format(frame_number)]
                
        f.create_dataset('/{}/img'.format(frame_number),data=narray)


def mask_file(h5file,frame_number=-1):
    '''
    masks all images in <h5file>
    '''
    logging.debug('masking file {}'.format(h5file))
    with h5py.File(h5file,'r+') as f:
        try:
            m = f.attrs['masked']
        except KeyError:
            m = 0
    
        if not m:
            frames = f.attrs['nframes']
            center = f.attrs['screen_center']
            radius = f.attrs['screen_radius']
            
            dataset = f.get('/0/img')
            narray = np.array(dataset)
            mask = get_mask(narray,center,radius)
            
            logging.info('Doing masking of file: {}'.format(h5file))

            if frame_number==-1:
                frames = utils.get_frames(h5file)
            else:
                frames = [frame_number]

            
            for i in frames:
                logging.debug('Doing masking of file: {}, frame: {}'.format(h5file,i))
                dataset = f.get('/{}/img'.format(i))
                narray = np.array(dataset)
                narray = ma.array(narray,mask = mask)
                
                #remove all cols and rows where every value is masked
                narray = narray[:, ~np.all(narray.mask, axis=0)].T
                narray = narray[:, ~np.all(narray.mask, axis=0)].T
               
                #fill masked values with zeros
                narray = ma.filled(narray,0)

                del f['/{}/img'.format(i)]
                #dataset[...] = narray
                
                f.create_dataset('/{}/img'.format(i),data=narray)

            f.attrs['dx'] = narray.shape[0]
            f.attrs['dy'] = narray.shape[1]
            #f.attrs['screen_center'] = [int(narray.shape[0]/2),int(narray.shape[1]/2)]
            if frame_number==-1:
                f.attrs['masked'] = 1
        else:
            logging.info('File {} is already masked'.format(h5file))
    return None
