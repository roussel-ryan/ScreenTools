import numpy as np
import numpy.ma as ma
import logging
import h5py
import matplotlib.pyplot as plt

from scipy import ndimage

from . import utils

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

def mask(h5file):
    '''
    masks all images in <h5file>
    '''

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

            for i in range(frames-1):
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
            f.attrs['masked'] = 1
        else:
            logging.info('File {} is already masked'.format(h5file))

def remove_array_background(array,plotting = False):
    #choose thresholding with iterative process
    #http://uspas.fnal.gov/materials/08UMD/Imaging.pdf
    
    signal_max = np.max(array,axis=None)
    threshold = np.linspace(0.0,signal_max,100)

    #fraction of peak derivative to cut off
    dthreshold = 0.25
    
    data = []
    for t in threshold:
        data.append(np.sum(np.where(array > t),axis=None))
    ndata = np.asfarray(data)

    
    d = np.gradient(ndata/np.max(ndata))
    dd = np.gradient(d)
    peak_dd_index = np.argmax(dd)

    
    nd = -d[peak_dd_index:]
    nthreshold = threshold[peak_dd_index:]
    dmax = np.max(nd)

    correct_index = np.argmax(nd/dmax < dthreshold)
    
    result = np.where(array > nthreshold[correct_index],array,0.0)
    
    if plotting:
        fig,ax = plt.subplots()
        ax.plot(threshold/np.max(threshold),ndata/np.max(ndata))
        ax2 = ax.twinx()           
        ax2.plot(threshold/np.max(threshold),d)
        ax2.plot(threshold/np.max(threshold),dd)
        ax.axvline(nthreshold[correct_index]/np.max(threshold),ls='--')

        fig2,ax3 = plt.subplots()
        ax3.imshow(result)
    return result

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
    
        
def remove_background(h5file):
    if not utils.get_attr(h5file,'background_removed'):
        logging.info('removing background from file {}'.format(h5file))
        with h5py.File(h5file,'r+') as f:
            frames = f.attrs['nframes']
            for i in range(frames-1):
                logging.debug('removing background from frame {}'.format(i))
                dataset = f['/{}/img'.format(i)]
                narray = remove_array_background(dataset[:])
                dataset[...] = narray
            f.attrs['background_removed'] = 1    
    return None
    
def filter_array(array,sigma=1,size=4):
    ''' apply a median filter and a gaussian filter to image'''
    ndata = ndimage.median_filter(array,size)
    return ndimage.gaussian_filter(ndata,sigma)
    
def filter(h5file,filter_high_res = False):
    '''filtering larger images is unnecessary'''
    if not utils.get_attr(h5file,'filtered'):
        with h5py.File(h5file,'r+') as f:
            if f.attrs['dx']*f.attrs['dy'] < 480*640 or filter_high_res:
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
        names = ['filtered','background_removed','masked']
        for name in names:
            try:
                del f.attrs[name]
            except KeyError:
                pass
