import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage

import logging
import h5py

import ScreenTools.utils as utils
import ScreenTools.file_processing as fp
import ScreenTools.image_processing as ip
import ScreenTools.plotting as plotting
import ScreenTools.analysis as analysis

def plot():
    ax = plotting.plot_screen('10_17/EYG2_9.h5')
    ax2 = plotting.plot_screen('10_17/EYG2_1.h5')
    ax3 = plotting.plot_screen('10_17/EYG2_5.h5')

    plotting.overlay_lineout('10_17/EYG2_9.h5',ax)
    plotting.overlay_lineout('10_17/EYG2_1.h5',ax2)
    plotting.overlay_lineout('10_17/EYG2_5.h5',ax3)
    
    logging.debug(analysis.fit_projection('10_17/EYG2_8.h5'))
    logging.debug(analysis.fit_projection('10_17/EYG2_1.h5',axis=0))
    logging.debug(analysis.fit_projection('10_17/EYG2_1.h5',axis=1))

def plot_EYG4():
    ax = plotting.plot_screen('10_17/EYG4_NOM.h5')
    plotting.overlay_lineout('10_17/EYG4_NOM.h5',ax)
    logging.debug(analysis.fit_projection('10_17/EYG4_NOM.h5'))
    logging.debug(analysis.fit_projection('10_17/EYG4_NOM.h5',axis=1))

def plot_beam_size():
    fig,ax = plt.subplots()
    
    data = []
    for i in range(1,10):
        fname = '10_17/EYG2_{}.h5'.format(i)
        stats = utils.get_attr(fname,'moments_stats')
        logging.debug(stats)
        sc = utils.get_attr(fname,'pixel_scale')
        data.append([i,*stats[3],*stats[4],sc])
    ndata = np.asfarray(data).T
    
    
    
    ax.errorbar(ndata[0]*0.1,ndata[1]*ndata[5],ndata[2]*ndata[5],fmt='-o',capsize=3,label='$\sigma_x$')
    ax.errorbar(ndata[0]*0.1,ndata[3]*ndata[5],ndata[4]*ndata[5],fmt='-o',capsize=3,label='$\sigma_y$')
    ax.set_ylabel('RMS beam size [m]')
    ax.set_xlabel('Focusing Quad Gradient [T/m]')
    ax.legend()
    return ax

def plot_energy_spread(ax2):
    f1 = '10_17/EYG6_{}.h5'
    f2 = '10_17/EYG7_{}.h5'

    #fig,ax = plt.subplots()
    ax = ax2.twinx()
    data = []
    for i in range(1,10):
        s1 = utils.get_attr(f1.format(i),'moments_stats')
        s2 = utils.get_attr(f2.format(i),'moments_stats')
    
        sE = np.sqrt(s2[3][0]**2 - s1[3][0]**2)
        dsE_s2 = s2[3][0] / sE
        dsE_s1 = -s1[3][0] / sE
        sError = np.sqrt(dsE_s1**2 * s1[3][1]**2 + dsE_s2**2 * s2[3][1]**2)
        
        data.append((i,sE,sError))
        
    ndata = np.asfarray(data).T
    ax.errorbar(ndata[0]*0.1,ndata[1]/np.max(ndata[1]),ndata[2]/np.max(ndata[1]),fmt='g-o',capsize=3,label='Energy Spread')
    ax.set_ylabel('Relative Energy Spread')
    ax.legend()
def main():
    #fname = '10_17/EYG4_NOM.dat'
    #fp.process_raw(fname,overwrite=False)
    #ip.mask('10_17/EYG4_NOM.h5')
    #ip.remove_background('10_17/EYG4_NOM.h5')
    base = '10_17/EYG7_{}.{}'
    
    for i in range(1,10):
        fname = base.format(i,'dat')
    
        fp.process_raw(fname,overwrite=True)
        #fp.add_current(fname)
 
    for i in range(1,10):
        fname = base.format(i,'h5')    
        ip.mask(fname)
        #ax = plotting.plot_screen(fname,image_number=1)
        ip.filter(fname)
        ip.remove_background(fname)
        
    
        analysis.calculate_moments(base.format(i,'h5'))
        
logging.basicConfig(level=logging.DEBUG)
#main()
#plot_energy_spread(plot_beam_size())
plotting.plot_screen('10_17/EYG6_8.h5',scaled=True)
logging.debug(analysis.calculate_moments('10_17/EYG6_8.h5'))
plt.show()

