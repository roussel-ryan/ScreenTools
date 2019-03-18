import numpy as np
import numpy.ma as ma
import logging
import h5py
import matplotlib.pyplot as plt

from scipy import ndimage

from .. import utils
from .. import plotting

def calculate_threshold(array,plotting = False):
    #choose thresholding with iterative process
    #http://uspas.fnal.gov/materials/08UMD/Imaging.pdf
    
    signal_max = np.max(array,axis=None)
    threshold = np.linspace(0.0,signal_max,500)

    #fraction of peak derivative to cut off
    dthreshold = 0.25
    
    data = []
    for t in threshold:
        data.append(np.sum(np.where(array > t),axis=None))
    ndata = np.asfarray(data)

    
    d = np.gradient(ndata/np.max(ndata))
    dd = np.gradient(d)
    peak_dd_index = np.argmax(dd)
    peak_dd = dd[peak_dd_index]
    ten_percent_level_peak_dd = 0.1*peak_dd 

    #cut AFTER peak of second dervative, at 10% level
    new_dd = dd[peak_dd_index:]
    new_index = np.argwhere(new_dd < ten_percent_level_peak_dd)[0] + peak_dd_index
    
    
    #correct_index = np.argmax(nd/dmax < dthreshold)
    correct_index = new_index    

    if plotting:
        fig,ax = plt.subplots()
        l1, = ax.plot(threshold/np.max(threshold),ndata/np.max(ndata),'g',label='Image fraction')
        ax.set_xlabel('Fractional threshold level')
        ax.set_ylabel('Remaining image fraction')
        ax2 = ax.twinx()           
        l2, = ax2.plot(threshold/np.max(threshold),d,label='Deriv(Image fraction)')
        l3, = ax2.plot(threshold/np.max(threshold),dd,label='DblDeriv(Image fraction)')
        l4 = ax.axvline(threshold[correct_index]/np.max(threshold),ls='--',label='Chosen threshold level')

        ls = [l1,l2,l3,l4]
        lbls = [l.get_label() for l in ls]
        ax.legend(ls,lbls,loc=7)
        
        result = np.where(array > threshold[correct_index],array,0.0)
        fig2,ax2 = plt.subplots(1,2)
        fig2.suptitle('Thresholding results at lvl: {}'.format(threshold[correct_index]))
        ax2[0].imshow(array)
        ax2[0].set_title('Pre-thresholding')
        ax2[1].imshow(result)
        ax2[1].set_title('Post-thresholding')
        
    return threshold[correct_index]

def plot_threshold(h5file,frame_number = 0):
    with h5py.File(h5file) as f:
        calculate_threshold(f['/{}/img'.format(frame_number)][:],plotting=True)
        
def set_threshold(h5file,level = 0,frame_number=-1):
    with h5py.File(h5file,'r+') as f:
        if frame_number == -1:
            frames = utils.get_frames(h5file)
            f['/'].attrs['global_threshold'] = level
        else:
            frames = [frame_number]

        if level:
            logging.info('Setting threshold level of all frames to {} in file {}'.format(level,h5file))
            for i in frames:
                datagrp = f['/{}'.format(i)]
                logging.debug('setting threshold of image {} to {}'.format(i,level))
                datagrp.attrs['threshold'] = level

        else:
            logging.info('Setting threshold level of all frames to based on calcuations in file {}'.format(h5file))
            for i in frames:                
                dataset = f['/{}/img'.format(i)][:]
                threshold = calculate_threshold(dataset)
                logging.debug('setting threshold of image {} to {}'.format(i,threshold)) 
                f['/{}'.format(i)].attrs['threshold'] = threshold

def apply_threshold(h5file,frame_number=-1):
    with h5py.File(h5file,'r+') as f:
        if frame_number==-1:
            frames = utils.get_frames(h5file)
        else:
            frames = [frame_number]
            
        for i in frames:
            dataset = f['/{}/img'.format(i)]
            threshold = f['/{}'.format(i)].attrs['threshold']
            logging.debug('applying threshold of {} to frame number {}'.format(threshold,i))
            dataset[...] = np.where(dataset[:] > threshold,dataset[:],0)
    
class ManualThresholdSelector:
    def __init__(self,filename,frame_number=-1):
        '''set threshold of file with a manual clicking that averages the height'''
        logging.info('Manual selection of threshold chosen using frame {}'.format(frame_number))
        self.filename = filename
        #load image array
        with h5py.File(self.filename) as f:
            if frame_number == -1:
                self.data = f['/0/img'][:]
                self.frame_number = -1
            else:
                self.data = f['/{}/img'.format(frame_number)][:]
                self.frame_number = frame_number
                
        fig,ax = plt.subplots()
        fig.suptitle('Click to select threshold points,SPACE to exit and set threshold\n,BACKSPACE to delete last point')
        self.im = ax.imshow(self.data)
        self.ax = ax
        
        self.points = [[0,0]]
        self.threshold = []
        self.line, = self.ax.plot(*self.points[0],'+')
        self.cid = self.ax.figure.canvas.mpl_connect('button_press_event',self.select_threshold)
        self.cid2 = self.ax.figure.canvas.mpl_connect('key_press_event', self.key_press)
        plt.show()

    def select_threshold(self,event):
        if event.inaxes!=self.ax: return
        self.points.append((event.xdata,event.ydata))
        logging.debug(self.points)
        self.threshold.append(self.data[int(event.ydata)][int(event.xdata)])
        logging.debug(self.threshold)
        self.line.set_data(*np.asfarray(self.points[1:]).T)

        self.data = np.where(self.data > np.max(np.asfarray(self.threshold)),self.data,0.0)
        self.im.set_data(self.data)
        self.ax.figure.canvas.draw()

    def remove_last_point(self):
        self.points = self.points[:-1]
        self.threshold = self.threshold[:-1]
        self.line.set_data(*np.asfarray(self.points[1:]).T)

        self.data = np.where(self.orig > np.max(np.asfarray(self.threshold)),self.orig,0.0)
        self.im.set_data(self.data)
        self.ax.figure.canvas.draw()
        
    def key_press(self,event):
        if event.key == ' ':
            set_threshold(self.filename,level=np.max(np.asfarray(self.threshold)),frame_number=self.frame_number)
            plt.close(self.ax.figure)

        if event.key == 'backspace':
            self.remove_last_point()
     
