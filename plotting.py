import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import h5py
import logging
from matplotlib.patches import Ellipse

from . import analysis
from . import utils
from . import current

class ScreenFigure(Figure):
    def __init__(self,*args,**kwargs):
        Figure.__init__(self,*args,**kwargs)
        self.filename = kwargs.pop('filename','')
        self.dataset = kwargs.pop('dataset','/img')
        self.frame_number = kwargs.pop('frame_number',0)
        self.scaled = kwargs.pop('scaled',False)

def create_screen_figure(**kwargs):
    fig = plt.figure(FigureClass=ScreenFigure,**kwargs)
    ax = fig.add_subplot(111)
    return fig,ax
        
def check_axes(axes):
    if axes:
        return axes
    else:
        fig,ax = create_screen_figure()
        return ax

def check_fig(fig):
    if isinstance(fig,ScreenFigure):
        return fig
    else:
        fig,ax = create_screen_figure()
        return fig

def plot_screen(filename,dataset = '/img',frame_number=0,fig=None,scaled=False):
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

    fig = check_fig(fig)
    ax = fig.axes[0]

    logging.info('plotting file {}, image # {}'.format(filename,frame_number))
    name = '/{}{}'.format(frame_number,dataset)
    with h5py.File(filename,'r') as f:
        data = f[name][:]
    shape = data.shape
    px_scale = utils.get_attr(filename,'pixel_scale')
    if scaled:
        ax.imshow(data,extent = [-int(shape[1]/2)*px_scale,int(shape[1]/2)*px_scale,-int(shape[0]/2)*px_scale,int(shape[0]/2)*px_scale],origin='lower')
        ax.set_aspect('equal')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        fig.scaled = True
    else:
        ax.imshow(data,origin='lower')
        fig.scaled = False
    ax.set_title('{} {}'.format(filename,name))
    fig.filename = filename
    fig.frame_number = frame_number
    fig.dataset = dataset
    
    #logging.debug(ax.screen_name)
    
    return fig

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

    l = len(data)
    for i in range(1,l):
        ax.plot(data[0],data[i])
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

def add_projection(figure,axis=0):
    '''
    overlay a projection (lineout) onto a image plot

    Inputs
    ------
    filename      h5 filename with screen info
    ax            matplotlib.axes object with screen image plotted
    image_number  frame number for plotting
    axis          0 for horizontal lineout, 1 for vertical axis lineout

    '''
    filename = figure.filename
    frame_number = figure.frame_number
    ax = figure.axes[0]
    
    lineout = analysis.get_projection(filename,frame_number,axis)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    logging.debug(xlim)
    logging.debug(ylim)
    if axis == 0:
        ax2 = ax.twinx()
        ax2.plot(np.linspace(*xlim,len(lineout)),lineout)
    elif axis == 1:
        ax2 = ax.twiny()
        ax2.plot(lineout,np.linspace(ylim[0],ylim[1],len(lineout)))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

def add_stats(figure):
    ax = figure.axes[0]
    moments = analysis.get_frame_moments(figure.filename,figure.frame_number)
    charge = current.get_frame_charge(figure.filename,figure.frame_number)

    std_x = np.sqrt(moments[1][0][0])
    std_y = np.sqrt(moments[1][1][1])
    
    if figure.scaled:
        data = (std_x,std_y)
        label = 'Image Stats\n $\sigma_x$: {:10.3} \n $\sigma_y$: {:10.3}\n Charge:\n'.format(*data)

    else:
        ps = utils.get_attr(figure.filename,'pixel_scale')
        data = (std_x / ps,std_y / ps)
        label = 'Image Stats\n $\sigma_x$: {:5.3g} \n $\sigma_y$: {:5.3g}\n Charge:\n'.format(*data)

    for ele in charge:
        label = label + '{:10.2g} nC\n'.format(ele*1e9)

    ax.text(0.95,0.95,label[:-1],horizontalalignment = 'right',verticalalignment = 'top',transform = ax.transAxes,backgroundcolor = 'white')
    
def add_ellipse(figure):
    filename = figure.filename
    frame_number = figure.frame_number
    ax = figure.axes[0]

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
        
    
