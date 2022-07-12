import numpy as np
import logging
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import RectangleSelector

from .. import utils
from .. import plotting

def rectangle_crop(fname,rectangle_extent,frame_number=-1):
    '''
    set all data elements outside of rectange_extent to zero
    for use with RectangleSelector extent attribute
    rectangle_extent = [x_min,x_max,y_min,y_max]
    '''
    with h5py.File(fname,'r+') as f:
        if frame_number == -1:
            frames = range(f.attrs['nframes'])
        else:
            frames = [frame_number]
        
        for frame in frames:
            dataset = f['/{}/img'.format(frame)]
            data = dataset[:]
            shape = data.shape
            mask = np.ones_like(data)

            for i in range(shape[0]):
                for j in range(shape[1]):
                    in_x = (i <= rectangle_extent[3]) and (i > rectangle_extent[2])
                    in_y = (j <= rectangle_extent[1]) and (j > rectangle_extent[0])

                    if in_x and in_y:
                        mask[i][j] = 0
                    
            new_array = np.ma.array(data,mask = mask)
            new_array = np.ma.filled(new_array,0)
            dataset[...] = new_array
            

class ManualCircularCrop:
    def __init__(self,filename,frame_number = 0):
        '''set all elements outside of selcected circle to zero'''
        
        logging.info('Manual cropping showing')
        self.filename = filename
        self.frame_number = frame_number
        #load image array
        with h5py.File(self.filename) as f:
            self.data = f['/{}/img'.format(frame_number)][:]
            self.orig = self.data
                          
        fig,ax = plt.subplots()
        fig.suptitle('Click to select circle middle,use arrow keys to expand/reduce radius')
        self.im = ax.imshow(self.data)
        self.size = self.data.shape[0]
        self.ax = ax
        
        self.center = np.array([0,0])
        self.radius = self.size / 2

        self.line, = self.ax.plot(*self.center,'+')
        self.circle = Circle(self.center,self.radius)
        self.circle.set_facecolor('none')
        self.circle.set_edgecolor('red')
        self.circle.set_linewidth(1)
        ax.add_artist(self.circle)
                          
        self.cid = self.ax.figure.canvas.mpl_connect('button_press_event',self.select_center)
        self.cid2 = self.ax.figure.canvas.mpl_connect('key_press_event', self.key_press)
        plt.show()

    def select_center(self,event):
        if event.inaxes!=self.ax: return
        self.center = np.array((event.xdata,event.ydata))
        self.update_circle()

    def increase_radius(self):
        self.radius = self.radius + 0.01*self.size
        self.update_circle()

    def decrease_radius(self):
        self.radius = self.radius - 0.01*self.size
        self.update_circle()
        
    def update_circle(self):
        self.circle.center = self.center[0],self.center[1]
        self.circle.radius = self.radius
        self.line.set_data(*self.center)
        self.ax.figure.canvas.draw()

    def crop(self,clip = True):
        shape = self.data.shape
        mask = np.ones_like(self.data)

        for i in range(shape[0]):
            for j in range(shape[1]):
                if clip:
                    if np.sqrt((j - self.center[0])**2 + (i - self.center[1])**2) < self.radius:
                        mask[i][j] = 0
                else:
                    if np.sqrt((j - self.center[0])**2 + (i - self.center[1])**2) > self.radius:
                        mask[i][j] = 0

        new_array = np.ma.array(self.data,mask = mask)
        new_array = np.ma.filled(new_array,0)
        with h5py.File(self.filename,'r+') as f:
            dataset = f['/{}/img'.format(self.frame_number)]
            dataset[...] = new_array
        
        
    def key_press(self,event):
        logging.debug(event.key)
        if event.key =='up':
            self.increase_radius()
        if event.key == 'down':
            self.decrease_radius()

        logging.debug(self.radius)
        if event.key == ' ':
            self.crop()
            plt.close(self.ax.figure)

        if event.key == 'c':
            self.crop(False)
            plt.close(self.ax.figure)
            
        #if event.key == 'backspace':
        #    self.remove_last_point()


class ManualRectangleCrop:
    def __init__(self,filename,frame_number = 0):
        '''set all elements outside of selcected circle to zero'''
        
        logging.info('Manual cropping showing')
        self.filename = filename
        self.frame_number = frame_number
        #load image array
        with h5py.File(self.filename) as f:
            self.data = f['/{}/img'.format(frame_number)][:]
            self.orig = self.data
                          
        fig,ax = plt.subplots()
        fig.suptitle('Click to select circle middle,use arrow keys to expand/reduce radius')
        self.im = ax.imshow(self.data)
        self.ax = ax

        self.RS = RectangleSelector(self.ax,self.line_select_callback,\
                                                    drawtype='box', useblit=True,\
                                                    button=[1,3],\
                                                    minspanx = 5, minspany=5,\
                                                    spancoords = 'data',\
                                                    interactive = True)
                                  
        self.cid2 = self.ax.figure.canvas.mpl_connect('key_press_event', self.key_press)
        plt.show()

    def line_select_callback(self,eclick, erelease):
        'eclick and erelease are the press and release events'
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        logging.debug("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
        logging.debug(" The button you used were: %s %s" % (eclick.button, erelease.button))

    def get_rectangle_extent(self):
        return self.RS.extents
        
    def key_press(self,event):
        logging.debug(event.key)
        if event.key == ' ':
            plt.close(self.ax.figure)
