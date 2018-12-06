import matplotlib.pyplot as plt
import numpy as np
import h5py
import logging

from . import analysis
from . import utils

def check_axes(axes):
    if axes:
        return ax
    else:
        fig,ax = plt.subplots()
        return ax

def plot_screen(filename,dataset = '/img',image_number=0,ax=None,scaled=False):
    ax = check_axes(ax)

    logging.info('plotting file {}, image # {}'.format(filename,image_number))
    name = '/{}{}'.format(image_number,dataset)
    with h5py.File(filename,'r') as f:
        data = f[name][:]
    shape = data.shape
    px_scale = utils.get_attr(filename,'pixel_scale')
    if scaled:
        ax.imshow(data,extent = [-int(shape[0]/2)*px_scale,int(shape[0]/2)*px_scale,-int(shape[1]/2)*px_scale,int(shape[1]/2)*px_scale])
        ax.set_aspect(shape[0]/shape[1])
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
    else:
        ax.imshow(data)
    ax.set_title('{} {}'.format(filename,name))
    return ax

def plot_current(filename,image_number=0,ax=None):
    ax = check_axes(ax)
    with h5py.File(filename,'r') as f:
        data = f['/{}/current'.format(image_number)][:]
    ax.plot(data[0],data[1])
    return ax
        
    
def overlay_lineout(filename,ax,image_number=0,axis=0):
    lineout = analysis.get_lineout(filename,image_number,axis)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if axis == 0:
        ax2 = ax.twinx()
        ax2.plot(lineout)
    elif axis == 1:
        ax2 = ax.twiny()
        ax2.plot(lineout,range(len(lineout)))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
