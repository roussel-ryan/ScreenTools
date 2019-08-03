import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.projections import register_projection
import matplotlib.gridspec as gridspec

import numpy as np
import h5py
import logging
from matplotlib.patches import Ellipse

from . import analysis
from . import utils
from . import current


class ScreenAxes(Axes):
    '''Subclassing matplotlib Axes object to add custom attributes'''
    name = 'ScreenAxes'    
    def __init__(self,*args,**kwargs):
        Axes.__init__(self,*args,**kwargs)
        self.filename = kwargs.pop('filename','')
        self.dataset = kwargs.pop('dataset','/img')
        self.frame_number = kwargs.pop('frame_number',0)
        self.scaled = kwargs.pop('scaled',False)

register_projection(ScreenAxes)

def subplots(col,row,**kwargs):
    ''' wrapper for creating suplots fig, axes with ScreenAxes object'''
    return plt.subplots(col,row,subplot_kw=dict(projection='ScreenAxes'),**kwargs)
        
def check_axes(axes):
    if axes:
        return axes
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='ScreenAxes')
        return ax

def check_fig(fig):
    if isinstance(fig,ScreenFigure):
        return fig
    else:
        fig,ax = create_screen_figure()
        return fig

def plot_screen(filename,raw=False,frame_number=0,ax=None,scaled=False,normalize = False, fast_plot=False,**kwargs):
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

    #fig = check_fig(fig)
    ax = check_axes(ax)

    logging.info('plotting file {}, image # {}'.format(filename,frame_number))
    if raw:
        dataset = '/raw'
    else:
        dataset = '/img'

    name = '/{}{}'.format(frame_number,dataset)
    with h5py.File(filename,'r') as f:
        data = f[name][:]

    if normalize:
        #set data to be normalized and shifted so min nonzero is 0.0
        #logging.info(np.min(data[np.nonzero(data)]))
        data = data.astype('int32')
        data = data - np.min(data[np.nonzero(data)])
        data = np.where(data > 0.0,data,0*data)
        data = data / np.max(data)
    
    shape = data.shape
    px_scale = utils.get_attr(filename,'pixel_scale')

    if fast_plot:
        data = data[::20,::20]
    
    if scaled:
        ax.imshow(data,extent = [-int(shape[1]/2)*px_scale,int(shape[1]/2)*px_scale,-int(shape[0]/2)*px_scale,int(shape[0]/2)*px_scale],origin='lower',**kwargs)
        ax.set_aspect('equal')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.scaled = True
        ax.px_scale = px_scale
    else:
        ax.imshow(data,origin='lower',**kwargs)
        ax.scaled = False
    ax.set_title('{} {}'.format(filename,name))
    ax.filename = filename
    ax.frame_number = frame_number
    ax.dataset = dataset
    
    #logging.debug(ax.screen_name)
    
    return ax

def plot_all_frames(filename):
    fig = plt.figure(figsize = (19.20,10.80))
    size = 5
    gs = gridspec.GridSpec(size,4*size)
    gs.update(wspace=0.0,hspace = 0.0)

    for i in range(4*size**2):
        ax = plt.subplot(gs[i])
        plot_screen(filename,frame_number=i,raw=False,ax = ax,fast_plot=True)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title('')
        #ax.set_aspect('equal')
    gs.tight_layout(fig)
    
def plot_background(filename):
    fig,ax = plt.subplots()
    with h5py.File(filename) as f:
        ax.imshow(f['/background'][:],origin='lower')
    return ax

def plot_current(filename,frame_number=0,ax=None):
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
        data = f['/{}/current'.format(frame_number)][:]
        charge = f['/{}'.format(frame_number)].attrs['charge']

    l = len(data)
    for i in range(1,l):
        ax.plot(data[0],data[i])

    label = ''
    for c in charge:
        label = label + '{:10.4g} nC\n'.format(c*1e9)

    ax.text(0.95,0.95,label[:-1],horizontalalignment = 'right',verticalalignment = 'top',transform = ax.transAxes,backgroundcolor = 'white')
    return ax

def plot_charge(filename,constraints=None):
    '''
    plot sequential and distribution charge data
    '''
    fig,axes = plt.subplots(2,1)
    data = []
    frames = utils.get_frames(filename,constraints)
    
    with h5py.File(filename) as f:
        for i in frames:
            data.append(f['/{}/'.format(i)].attrs['charge'])

    data = np.asfarray(data).T
    for ele in data:
        axes[0].plot(ele)

        h,be = np.histogram(ele,bins='auto')
        bc = (be[1:]+be[:-1]) / 2
        axes[1].plot(bc,h)

def add_projection(ax,axis=0,**kwargs):
    '''
    overlay a projection (lineout) onto a image plot

    Inputs
    ------
    filename      h5 filename with screen info
    ax            matplotlib.axes object with screen image plotted
    image_number  frame number for plotting
    axis          0 for horizontal lineout, 1 for vertical axis lineout

    '''
    filename = ax.filename
    frame_number = ax.frame_number
    extent = ax.get_images()[0].get_extent()
    
    lineout = analysis.get_projection(filename,frame_number,axis)
    lineout = lineout / np.max(lineout)
    xlim = extent[:2]
    ylim = extent[2:]
    #xlim = ax.get_xlim()
    #ylim = ax.get_ylim()

    logging.debug(xlim)
    logging.debug(ylim)
    if axis == 0:
        ext = ylim[1] - ylim[0]
        ax.plot(np.linspace(*xlim,len(lineout)),lineout*0.25*ext + ylim[0],**kwargs)
    elif axis == 1:
        ext = xlim[1] - xlim[0]
        ax.plot(lineout*0.25*ext + xlim[0],np.linspace(*ylim,len(lineout)),**kwargs)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

def add_stats(ax):
    filename = ax.filename
    frame_number = ax.frame_number
    
    moments = analysis.get_frame_moments(filename,frame_number)
    charge = current.get_charge(filename,frame_number)

    std_x = np.sqrt(moments[1][0][0])
    std_y = np.sqrt(moments[1][1][1])
    
    if ax.scaled:
        data = (std_x,std_y)
        label = 'Image Stats\n $\sigma_x$: {:10.3} \n $\sigma_y$: {:10.3}\n Charge:\n'.format(*data)

    else:
        ps = utils.get_attr(figure.filename,'pixel_scale')
        data = (std_x / ps,std_y / ps)
        label = 'Image Stats\n $\sigma_x$: {:5.3g} \n $\sigma_y$: {:5.3g}\n Charge:\n'.format(*data)

    for ele in charge[0]:
        label = label + '{:10.4g} nC\n'.format(ele*1e9)

    ax.text(0.95,0.95,label[:-1],horizontalalignment = 'right',verticalalignment = 'top',transform = ax.transAxes,backgroundcolor = 'white')
    
def add_ellipse(ax):
    filename = ax.filename
    frame_number = ax.frame_number
    #ax = figure.axes[0]

    logging.debug('calcuating ellipse with file {} and frame number {}'.format(filename,frame_number))
    
    moments = analysis.get_frame_moments(filename,frame_number)
    lambda_,v = np.linalg.eig(moments[1])
    logging.debug(lambda_)
    angle = np.arctan(v[0][1]/v[0][0])
    
    ell = Ellipse(xy=[moments[0][0],moments[0][1]],\
                  width = lambda_[0]**(0.5)*2,\
                  height = lambda_[1]**(0.5)*2,\
                  angle = -np.rad2deg(angle)) 
    ell.set_facecolor('none')
    ell.set_edgecolor('red')
    ell.set_linewidth(1)
    ax.add_artist(ell)

def add_meanline(ax,axis=0,test=False,**kwargs):
    filename = ax.filename
    frame_number = ax.frame_number
    extent = ax.get_images()[0].get_extent()
    try:
        px_scale = ax.px_scale
    except:
        px_scale = 1
    xlim = extent[:2]
    ylim = extent[2:]
    xext = xlim[1] - xlim[0]
    yext = ylim[1] - ylim[0]

    line = analysis.get_mean_line(filename,frame_number,axis=axis)
    line = line
    logging.debug(extent)
    
    if test: fig,ax2 = plt.subplots()

    if axis == 0:
        ext = ylim[1] - ylim[0]
        if test:
            ax2.plot(np.linspace(*xlim,len(line)),line*px_scale + ylim[0],**kwargs)
        ax.plot(np.linspace(*xlim,len(line)),line*px_scale + ylim[0],**kwargs)
    elif axis == 1:
        ext = xlim[1] - xlim[0]
        if test:
            ax2.plot(line*px_scale + xlim[0],np.linspace(*ylim,len(line)),**kwargs)
        ax.plot(line*px_scale + xlim[0],np.linspace(*ylim,len(line)),**kwargs)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    #if test:
    #    fig,ax2 = plt.subplots()
    #    logging.info(line)
    #    if axis:
    #        ax2.plot(line[1],line[0])
    #    else:
    #        ax2.plot(line[0],line[1])

    #ax.set_ylim(ylim)
    #ax.set_xlim(xlim)

def add_contours(ax,levels=10):
    filename = ax.filename
    frame_number = ax.frame_number
    data = utils.get_frame_image(filename,frame_number)
    c = ax.contour(data,levels=levels,linewidth=10)
    return c
    
def plot_charge_histogram(filename,constraints=None):
    charge = []
    frames = utils.get_frames(filename,constraints)
    with h5py.File(filename) as f:
        for i in frames:
            charge.append(f['/{}'.format(i)].attrs['charge'])

    charge = np.asfarray(charge).T
    fig,ax = plt.subplots()
    for ele in charge:
        h,bine = np.histogram(ele,bins='auto')
        binc = (bine[1:] + bine[:-1]) / 2
        ax.plot(binc*1e9,h)
    ax.set_xlabel('Charge [nC]')
    ax.set_ylabel('N frames')
        
    
