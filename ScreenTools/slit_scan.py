import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from os import listdir
from os.path import isfile, join

from scipy import ndimage

import logging
import h5py

from . import utils
import ScreenTools.file_processing as fp
import ScreenTools.image_processing as ip
import ScreenTools.plotting as plotting
import ScreenTools.analysis as analysis
from . import current
from .processing import thresholding

def process_files(path):
    #fp.process_folder(path,same_screen=True)
    files = utils.get_files(path,'.h5')
    for file in files:
        ip.reset_file(file)
        ip.mask_file(file)
        ip.filter_file(file,filter_high_res = True)
    return files
        
def plot_files(path,overlay=True):
    files = utils.get_files(path,'.h5')
    for file in files:
        ax = plotting.plot_screen(file)
        if overlay:
            plotting.overlay_lineout(file,ax)

def get_stepper_locations(path):
    '''return sorted array of stepper motor locations in terms of steps'''
    files = utils.get_files(path,'.h5')
    locations = []
    for ele in files:
        locations.append(int(ele.split('StepperScan_')[1].split('.h5')[0]))
    locations = np.sort(np.array(locations))
    return locations
        
def get_list_middle(locations):
    length = len(locations)
    logging.debug(length)
    if length % 2 == 0:
        logging.warning('number of stepper locations should be odd! returning next to middle')
        return locations[int((length-1) / 2)]
    else:
        return locations[int((length-1) / 2)]

def threshold_files(path):
    '''get proper threshold for middle location and apply the threshold to the rest of the files'''

    logging.info('Thresholding slit scan files in path {}'.format(path))
    middle_location = get_stepper_middle(get_stepper_locations(path))
    middle_file = '{}StepperScan_{}.h5'.format(path,middle_location)
    ip.set_threshold(middle_file)

    #get avg threshold from file
    with h5py.File(middle_file) as f:
        logging.debug('Getting avg threshold from {}'.format(middle_file))
        nframes = f['/'].attrs['nframes']
        thresholds = []
        for i in range(nframes-1):
            thresholds.append(f['/{}'.format(i)].attrs['threshold'])
        threshold = np.mean(np.asfarray(thresholds))
        logging.debug('Threshold found to be {}'.format(threshold))
        
    #set this threshold to every file
    files = utils.get_files(path,'.h5')
    for file in files:
        ip.threshold_file(file,level=threshold)
    
def map_slit_locations(path,slit_width,steps_to_m,plotting=False):
    '''
    get all of the slit locations for plotting and further processing,zero point is avg position
    '''
    #get locations in terms of steps
    locations = get_stepper_locations(path)

    #location in terms of meters
    locations = (locations - np.mean(locations)) * steps_to_m

    if plotting:
        fig,ax = plt.subplots()
        for loc in locations:
            ax.axvline(loc - slit_width/2)
            ax.axvline(loc + slit_width/2)
        ax.axvline(get_list_middle(locations),ls='--')
                        
    return locations

def map_projections(path,slit_locations,slit_width,L,condition = current.mean_charge,plotting=False):
    locations = get_stepper_locations(path)

    #get the first file and assume that there is no beam in it
    zero_file = '{}StepperScan_{}.h5'.format(path,int(locations[0]))
    zero_proj = analysis.get_averaged_projection(zero_file,condition)

    projections = []
    for pos in locations:
        proj = analysis.get_averaged_projection('{}StepperScan_{}.h5'.format(path,int(pos)),condition) - zero_proj
        projections.append([pos,proj])

    #set middle distribution mean to zero point in real space
    middle_dist = get_list_middle(projections)[1]
    x = np.arange(len(middle_dist))
    mean = np.sum(x*middle_dist)/np.sum(middle_dist)
    px_scale = utils.get_attr(zero_file,'pixel_scale')

    #do the transformation from final screen projection to x'
    xp_data = []
    for ele,loc in zip(projections,slit_locations):
        xp = px_scale*(x - mean) + loc
        xp_data.append([xp,ele[1]])

    #interpolate each dataset to fit on a unifromly spaced grid for use in imshow
    xp_grid = []
    n_pts = 1000

    #get global max and min for extraoplation
    ma = 0.0
    mi = 0.0

    threshold = []
    for dataset in xp_data:
        x = dataset[0]
        if np.min(x) < mi:
            mi = np.min(x)
        if np.max(x) > ma:
            ma = np.max(x)

        #threshold.append(thresholding.calculate_threshold(dataset[1]))
    n = np.linspace(mi,ma,n_pts)

    #interpolate functions such that they are spaced evenly
    interp = []
    for dataset in xp_data:
        interp.append(np.interp(n,dataset[0],dataset[1]))
        
    
    if plotting:
        fig,ax = plt.subplots()
        fig2,ax2 = plt.subplots()
        fig3,ax3 = plt.subplots()
        for ele,loc in zip(projections,slit_locations):
            x0 = px_scale*(x-mean)
            ax.plot(x0,ele[1])
            ax.axvline(loc - slit_width/2)
            ax.axvline(loc + slit_width/2)

        for dataset in xp_data:
            #xp = np.where(dataset[1] > t,dataset[1],0.0)
            ax2.plot(dataset[0],dataset[1])

        for inter in interp:
            ax3.plot(n,inter)
            
        ax.axvline(get_list_middle(slit_locations),ls='--')

    return n,interp

    
        
def reconstruct_phase_space(path,drift_length,slit_width,m_per_step,plotting=True):
    files = utils.get_files(path,'.h5')
    
    #create image of reconstructed phase space, pixels here refers to image pixels

    #x location is determined by the slit location
    #screen image projections along the x-axis are used as the y-axis in the reconstruction
    #For the phase space reconstruction y-axis y = (x_screen - x_slit_location) / L

    
    # x manipulations

    #get slit_locations in terms of stepper steps
    #locations = get_stepper_locations(path)
    
    #x_size = 1000
    #x_buffer = 0.2
    #pixels_per_step = int(x_size / stepper_stroke*(1+x_buffer))
    #x_scale = 1 / (pixels_per_step * m_per_step)
    #slit_width_in_steps = slit_width * m_per_step
    #slit_width_in_pixels = int(slit_width_in_steps * pixels_per_step)
    
    #logging.info('Steps per pixel: {}'.format(pixels_per_step))
    #logging.info('m per step: {}'.format(m_per_Step))
    #logging.info('m per pixel: {}'.format(x_scale))
    #logging.info('Slit width in steps: {}'.format(slit_width_in_steps))
    #logging.info('Slit width in image pixels: {}'.format(slit_width_in_pixels))
    
    #pixel_stepper_pos = (stepper_pos - min_pos) * pixels_per_step + (int(x_size/2) - int(stepper_stroke/2) * pixels_per_step) 
    #pixel_stepper_pos = pixel_stepper_pos.astype(int)
    #logging.debug(pixel_stepper_pos)

    #get slit locations in meters
    slit_locations = map_slit_locations(path,slit_width,m_per_step)
    
    #convert from meters to pixels
    x_pixels = 1000
    stroke = slit_locations[-1] - slit_locations[0]
    x_size = stroke * 1.2
    pixels_per_m = int(x_pixels / x_size)
    slit_width_in_pixels = int(pixels_per_m * slit_width)

    slit_locations_in_pixels = slit_locations*pixels_per_m + x_pixels / 2
    slit_locations_in_pixels = slit_locations_in_pixels.astype(int)
    
    #get_properly scaled/shifted projections with background removed
    #add plotting = True for plot
    xp,density = map_projections(path,slit_locations,slit_width,drift_length)
    y_pixels = len(xp)
    y_size = xp[-1] - xp[0]

    reconstruction_image = np.zeros((x_pixels,y_pixels))

    for pixel_location,den in zip(slit_locations_in_pixels,density):
        for i in range(slit_width_in_pixels):
            reconstruction_image[pixel_location + i - int(slit_width_in_pixels / 2)] = np.flipud(den)

    #do thresholding for reconstructed array
    threshold = thresholding.calculate_threshold(reconstruction_image)
    reconstruction_image = np.where(reconstruction_image > threshold,reconstruction_image,0.0)

    #create image box
    ib = ip.ImageBox()
    ib.set_size((x_size,y_size))

    #center the image box
    ib = analysis.center_distribution(reconstruction_image.T,ib)
    
    if plotting:
        fig,ax = plt.subplots()
        ax.imshow(reconstruction_image.T,origin='lower',aspect='auto',extent = ib.get_extent())
        ax.set_ylabel('x\' [rad]')
        ax.set_xlabel('x [m]')
        ax.axvline(0.0)
        ax.axhline(0.0)
    return reconstruction_image.T,ib

def calculate_emittance(image,image_box,plotting=True):
    stats = analysis.calculate_ellipse(image,image_box)
    logging.debug(stats)

    y_size,x_size = image.shape
    ext = image_box.get_extent()
    
    midpt_scaled = image_box.center

    #scaling for eigenvectors
    ev = stats[2][0]

    if plotting:
        fig,ax = plt.subplots()
        ax.imshow(image,extent=ext,origin='lower')
        ax.set_aspect('auto')

        for j in range(1,4):
            ell = Ellipse(xy=stats[0],\
                          width=stats[1][0]**(0.5)*j*2,\
                          height=stats[1][1]**(0.5)*j*2,\
                          angle = np.rad2deg(-np.arctan(ev[1]/ev[0])))
            ell.set_facecolor('none')
            ell.set_edgecolor('red')
            ell.set_linewidth(1)
            ax.add_artist(ell)
        ax.axhline(0.0)
        ax.axvline(0.0)

    #get real moments, NOTE: need to transpose image array to get propermoments 
    moments = analysis.get_array_moments(image,image_box)    

    logging.info('rms_x: {} mm'.format(moments[1][0][0]**0.5*1000))
    logging.info('rms_xp: {} mrad'.format(moments[1][1][1]**0.5*1000))
    logging.info('Slope: {}'.format(np.arctan(ev[1]/ev[0])))
    logging.info('Geometric Emittance: {} mm.mrad'.format(10**6 * 13.2*np.sqrt(np.linalg.det(moments[1]))))
        
