#YAG class to store YAG data from AWA can be extended in the future??

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import h5py
from . import gaussfitter2

class YAG:
    def __init__(self,name,filename,**kwargs):
        self.name = name
        self.dat_filename = filename
        self.stat_filename = '{}.stat'.format(self.dat_filename.split('.')[0])
        self.import_images(**kwargs)

        self.stats = {}
        
    def import_stat(self):
        if os.path.isfile(self.stat_filename):
            with open(self.stat_filename,'r') as f: 
                self.center = np.round(np.asfarray(f.readline().strip().split(' ')))
                self.radius = np.round(float(f.readline().strip()))
                self.scale = float(f.readline().strip())

    def import_images(self,header_size=6,order_type='C'):
        '''
        This function reads in image data
        It assumes the first three bits are the 
        Horizontal size (X), Vertical size (Y),
        and number of frames (Nframes) respectively
    
        count=-1 -> reads all data
        sep='' -> read file as binary
        header_size=6 for old data aquisition (at AWA)  
        header_size=3 for new python data aquisition (at AWA)
 
        header info vert/horiz pixels and number of frames
        '''
        logging.info('reading data from file: {}'.format(self.dat_filename))
        data    = np.fromfile(self.dat_filename, dtype=np.uint16)
        dx = int(data[1])
        dy = int(data[0])

        if header_size==6:
            Nframes = int(data[2])+1
            length  = dx*dy*Nframes   
            n = header_size# + 1 
            images  = data[n:]
        else:
            Nframes = int(data[2])
            length  = dx*dy*Nframes   
            n = header_size + 1 
            images  = data[n:]

        logging.info((dx,dy,length,data[:10]))
        logging.info(images.shape)
            
        images = np.reshape(images,(-1, dx, dy), order=order_type)
        logging.info('Done reading images')

            
        
        
class Image:
    def __init__(self,array,dx,dy,center=[],radius=0.0,scale=1.0):
        self.array = array
        self.dx = dx
        self.dy = dy
        
        self.screen_center = center
        self.screen_radius = radius
        
    def trim_array(self):
        '''trim away cells to reduce overall array size'''
        #do for one axis
        empty_test = [ele.count() for ele in self.array]
        self.array = self.array[np.nonzero(empty_test)].T
        
        #again for the other axis
        empty_test = [ele.count() for ele in self.array]
        self.array = self.array[np.nonzero(empty_test)].T
        
        logging.info(self.array.shape)
        
    def get_lineout(self,axis='x'):
        if axis == 'x':
            return np.sum(self.array,axis=0)
        else:
            return np.sum(self.array,axis=1)

    def mask_array(self):
        #start by making a mask array
        self.mask = np.ones_like(self.array)
        shape = self.mask.shape
        mask_radius = 0.9*self.screen_radius
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                if np.sqrt((j - self.screen_center[0])**2 + (i - self.screen_center[1])**2) < mask_radius:
                    self.mask[i][j] = 0
        
        self.array = np.ma.masked_array(self.array,self.mask)
        
    def subtract_background(self,plotting = False):
        #choose thresholding with iterative process
        #http://uspas.fnal.gov/materials/08UMD/Imaging.pdf
        
        signal_max = np.max(self.array,axis=None)
        threshold = np.linspace(0.0,signal_max,100)
        
        data = []
        for t in threshold:
            data.append(np.sum(np.where(self.array > t),axis=None))
        ndata = np.asfarray(data)
        
        d = np.gradient(ndata/np.max(ndata))
        dd = np.gradient(d)
        correct_index = np.argmax(dd)
        
        
        if plotting:
            fig,ax = plt.subplots()
            ax.plot(threshold/np.max(threshold),ndata/np.max(ndata))
            ax2 = ax.twinx()           
            ax2.plot(threshold/np.max(threshold),d)
            ax2.plot(threshold/np.max(threshold),dd)
            ax.axvline(threshold[correct_index]/np.max(threshold),ls='--')
        
        self.array = self.array - threshold[correct_index]
        
    def plot(self,ax='',lineout=False,x_lineout=False,y_lineout=False,lineout_height = 0.2):
        if not ax:
            fig,ax = plt.subplots()

        ax.imshow(self.array)
        x_color = 'r'
        y_color = 'g'
        
        if lineout:
            x_lineout = True
            y_lineout = True
            
            
        if x_lineout:
            ax2 = ax.twinx()
            tmp = self.get_lineout('x')
            line = tmp - np.min(tmp)
            ax2.plot(np.arange(0,len(line)),lineout_height*line/np.max(line),x_color)
            ax2.set_ylim(0.0,1.0)
        if y_lineout:
            ax3 = ax.twiny()
            tmp = self.get_lineout('y')
            line = tmp - np.min(tmp)
            ax3.plot(lineout_height*line/np.max(line),np.arange(0,len(line)),y_color)
            ax3.set_xlim(0.0,1.0)
            
        #ax.set_xlim(0,self.dy)
        #ax.set_ylim(0,self.dx)
            
        
        return ax

    def fit_gaussian(self):
        input_guesses = [0.0,np.max(self.array),*self.screen_center,\
            self.screen_radius/2.0,self.screen_radius/2,0.0]
        fit = gaussfitter2.gaussfit(self.array,params=input_guesses)
        logging.info(fit)



def import_data(filename,header_size=6,order_type='C'):
    ''' import image data into h5 format for more efficient computing'''
    logging.info('reading data from file: {}'.format(filename))
    data    = np.fromfile(filename, dtype=np.uint16)
    dx = int(data[1])
    dy = int(data[0])
    
    if header_size==6:
        nframes = int(data[2])+1
        length  = dx*dy*nframes   
        n = header_size# + 1 
        images  = data[n:]
    else:
        nframes = int(data[2])
        length  = dx*dy*nframes   
        n = header_size + 1 
        images  = data[n:]

    images = np.reshape(images,(-1, dx, dy), order=order_type)

    logging.info(np.max(images,axis=None))
    
    #now send all the images to preallocated datasets
    with h5py.File('{}.h5'.format(filename.split('.')[0]),'w') as f:
        #attach metadata to main group
        f.attrs['dx'] = dx
        f.attrs['dy'] = dy
        f.attrs['nframes'] = nframes
        
        for i in range(len(images)):
            f.create_dataset('im_{}'.format(i),data=images[i],dtype=np.uint16)
    
    logging.info('Done importing')    
    return None 
