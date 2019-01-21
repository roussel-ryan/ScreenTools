import matplotlib.pyplot as plt
import numpy as np
import h5py
import logging
from matplotlib.patches import Ellipse

from . import analysis
from . import utils

def check_axes(axes):
    if axes:
        return axes
    else:
        fig,ax = plt.subplots()
        return ax

def plot_screen(filename,dataset = '/img',image_number=0,ax=None,scaled=False):
    '''
    Plot screen image for a given frame
    
    Inputs
    ------
    filename         h5 file with screen information
    dataset          specify what type of image to plot (either /img or /raw
    image_number     frame number for image
    ax               matplotlib.axes object
    scaled           if True, scale image using given pixel scale

    Outputs
    -------
    matplotlib.axes object with image plot

    '''

    ax = check_axes(ax)

    logging.info('plotting file {}, image # {}'.format(filename,image_number))
    name = '/{}{}'.format(image_number,dataset)
    with h5py.File(filename,'r') as f:
        data = f[name][:]
    shape = data.shape
    px_scale = utils.get_attr(filename,'pixel_scale')
    if scaled:
        ax.imshow(data,extent = [-int(shape[1]/2)*px_scale,int(shape[1]/2)*px_scale,-int(shape[0]/2)*px_scale,int(shape[0]/2)*px_scale])
        ax.set_aspect('equal')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
    else:
        ax.imshow(data)
    ax.set_title('{} {}'.format(filename,name))
    return ax

def plot_current(filename,image_number=0,ax=None):
    '''
    plot current data from ICT for given frame
    
    Inputs
    ------
    filename        h5 file with screen information
    image_number    frame number
    ax              matplotlib.axes object for plotting

    Output
    ------
    matplotlib.axes object with current plot

    '''


    ax = check_axes(ax)
    with h5py.File(filename,'r') as f:
        data = f['/{}/current'.format(image_number)][:]
    ax.plot(data[0],data[1])
    return ax

def plot_charge(filename):
    '''
    plot sequential and distribution charge data
    '''
    fig,axes = plt.subplots(2,1)
    data = []
    with h5py.File(filename) as f:
        for i in range(f['/'].attrs['nframes']-1):
            data.append(f['/{}/'.format(i)].attrs['charge'])

    data = np.asfarray(data)
    axes[0].plot(data)

    h,be = np.histogram(data,bins='auto')
    bc = (be[1:]+be[:-1]) / 2
    axes[1].plot(bc,h)

def overlay_projection(filename,ax,image_number=0,axis=0):
    '''
    overlay a projection (lineout) onto a image plot

    Inputs
    ------
    filename      h5 filename with screen info
    ax            matplotlib.axes object with screen image plotted
    image_number  frame number for plotting
    axis          0 for horizontal lineout, 1 for vertical axis lineout

    '''
    
    lineout = analysis.get_lineout(filename,image_number,axis)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if axis == 0:
        ax2 = ax.twinx()
        ax2.plot(np.linspace(*xlim,len(lineout)),lineout)
    elif axis == 1:
        ax2 = ax.twiny()
        ax2.plot(lineout,range(len(lineout)))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)



    
