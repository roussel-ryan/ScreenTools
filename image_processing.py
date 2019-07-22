import numpy as np
import numpy.ma as ma
import logging
import h5py
import matplotlib.pyplot as plt
from scipy import ndimage

from .processing import thresholding
from .processing import masking
from .processing import cropping
from .processing import background
from . import utils



class ImageBox:
    ''' 
    class to store and transform scaling for images for pixel->real space
    '''
    def __init__(self):
        self.center = np.asfarray([0,0])
        self.size = np.asfarray([1,1])

    def set_center(self,center):
        self.center = center

    def set_size(self,size):
        self.size = size

    def set_bbox(self,coords):
        ''' coords in the form [[x0,y0],[x1,y1]]
        where 
        (x0,y0) is bottom left and
        (x1,y1) is bottom right
        '''
        if not type(coords) == np.ndarray:
            coords = np.asfarray(coords)
        self.center = np.asfarray(((coords[0][0] + coords[1][0])/2,(coords[0][1] + coords[1][1])/2))
        self.size = np.asfarray((coords[1][0]-coords[0][0],coords[1][1] - coords[0][1]))

    def check_type(self):
        if not type(self.center) == np.ndarray:
            self.center = np.asfarray(self.center)
        if not type(self.size) == np.ndarray:
            self.size = np.asfarray(self.size)
    
    def get_bbox(self):
        self.check_type()
        v0 = self.center - self.size / 2
        v1 = self.center + self.size / 2
        return np.vstack(v0,v1)

    def get_extent(self):
        self.check_type()
        v0 = self.center - self.size / 2
        v1 = self.center + self.size / 2
        return [v0[0],v1[0],v0[1],v1[1]]

        
def quick_process_file(files):
    for f in files:
        mask_file(f)
        filter_file(f)
    for f in files:
        threshold_file(f,manual=True)

def quick_process_frame(f,frame_number):
    mask_frame(f,frame_number)
    filter_frame(f,frame_number)
    threshold_frame(f,frame_number=frame_number,manual=True)
    
def mask(*args,**kwargs):
    masking.mask(*args,**kwargs)
    
def remove_background(h5file,frame_number=-1,overwrite=False):
    
    with h5py.File(h5file,'r+') as f:
        bkgrnd = f['background'][:]
        if frame_number == -1:
            frames = utils.get_frames(h5file)
            logging.info('removing background from all frames file:{}'.format(h5file))
        else:
            frames = [frame_number]
            logging.info('removing background from frame {}, file:{}'.format(frame_number,h5file))
            
        for frame in frames:
            dset = f['/{}/img'.format(frame)]
            try:
                bkgrnd_removed = dset.attrs['background_removed']
            except KeyError:
                bkgrnd_removed = 0
            if (not bkgrnd_removed) or overwrite:
                dset[...] = background.subtract_background(dset[:],bkgrnd)
                dset.attrs['background_removed'] = 1


def threshold(h5file,frame_number=-1,level=0):
    if level == 0:
        m = thresholding.ManualThresholdSelector(h5file,frame_number).threshold
    else:
        m = level

    frames = utils.check_frame_number(h5file,frame_number)

    apply_threshold(h5file,frames,m)

def apply_threshold(h5file,frames,level):
    with h5py.File(h5file,'r+') as f:
        for ele in frames:
            try:
                grp = f['/{}'.format(ele)]
                current_level = grp.attrs['threshold']
            except KeyError:
                current_level = 0

            if level > current_level:
                dataset = grp['img']
                data = dataset[:]
                dataset[...] = np.where(data > level,data,0)
                grp.attrs['threshold'] = level
    

def get_rect_crop(h5file,frame_number):
    c = cropping.ManualRectangleCrop(h5file,frame_number)
    return c.get_rectangle_extent()


def crop_frame(h5file,extent = '',frame_number=0):
    #add ability to recall cropping OR get cropped rectangle and use rectangle in the future to keep settings -Ryan
    if extent == '':
        extent = get_rect_crop(h5file,frame_number)
    
    cropping.rectangle_crop(h5file,extent,frame_number)

def crop_file(h5file,extent=''):
    if extent == '':
        extent = get_rect_crop(h5file,0)
        
    cropping.rectangle_crop(h5file,extent,-1)

        
        
def filter_array(array,sigma=1,size=4):
    ''' apply a median filter and a gaussian filter to image'''
    ndata = ndimage.median_filter(array,size)
    return ndata
    #return ndimage.gaussian_filter(ndata,sigma)

def filter_frame(h5file,frame_number):
    with h5py.File(h5file,'r+') as f:
        dataset = f['/{}/img'.format(frame_number)]
        try:
            filtered_already = dataset.attrs['filtered']
        except KeyError:
            filtered_already = False
            
        if not filtered_already:
            narray = filter_array(dataset[:])
            dataset[...] = narray
            dataset.attrs['filtered'] = 1
    
def filter_file(h5file):
    '''filtering'''
    if not utils.get_attr(h5file,'filtered',dataset='/img/0'):
        logging.info('applying median filter to file {}'.format(h5file))

        for i in utils.get_frames(h5file):
            logging.debug('filtering frame {}'.format(i))
            filter_frame(h5file,i)
                
def reset(h5file,frame_number=-1):
    frames = utils.check_frame_number(h5file,frame_number)
    logging.debug('resetting file {}, frames {}'.format(h5file,frames))
    
    with h5py.File(h5file,'r+') as f:
        for ele in frames:
            logging.debug('resetting frame {}'.format(ele))
            grp = f['/{}'.format(ele)]
            del grp['img']
            raw = grp['raw'][:]

            grp.create_dataset('img'.format(frame_number),data=raw)
            grp.attrs['threshold'] = 0
            grp.attrs['filtered'] = 0
            grp.attrs['masked'] = 0
        
