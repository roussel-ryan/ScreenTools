import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from os import listdir
from os.path import isfile, join

from scipy import ndimage

import logging
import h5py

import ScreenTools.utils as utils
import ScreenTools.file_processing as fp
import ScreenTools.image_processing as ip
import ScreenTools.plotting as plotting
import ScreenTools.analysis as analysis


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
    '''return sorted array of stepper motor locations'''
    files = utils.get_files(path,'.h5')
    locations = []
    for ele in files:
        locations.append(int(ele.split('StepperScan_')[1].split('.h5')[0]))
    locations = np.sort(np.array(locations))
    return locations
        
def get_stepper_middle(locations):
    length = len(locations)
    if length % 2:
        logging.warning('number of stepper locations should be odd! returning next to middle')
        return locations[int(length / 2)]
    else:
        return locations[length / 2]

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
    
    
def reconstruct_phase_space(path,drift_length,slit_width,steps_to_m,plotting=True):
    files = utils.get_files(path,'.h5')
    logging.debug(files)
    reconstruction_image = []
    
    stepper_pos = get_stepper_locations(path)
    
    max_pos = np.max(stepper_pos)
    min_pos = np.min(stepper_pos)
    stepper_stroke = max_pos - min_pos
    stepper_center = min_pos + int(stepper_stroke / 2)
    
    #create image of reconstructed phase space, pixels here refers to image pixels
    x_size = 1000
    x_buffer = 0.2
    steps_to_pixels = int(stepper_stroke*(1+x_buffer) / x_size)
    x_scale = steps_to_pixels / steps_to_m
    slit_width_in_steps = slit_width * steps_to_m
    slit_width_in_pixels = int(slit_width_in_steps / steps_to_pixels)
    
    
    logging.info('Steps per pixel: {}'.format(steps_to_pixels))
    logging.info('m per step: {}'.format(1/steps_to_m))
    logging.info('m per pixel: {}'.format(x_scale))
    logging.info('Slit width in steps: {}'.format(slit_width_in_steps))
    logging.info('Slit width in image pixels: {}'.format(slit_width_in_pixels))
    
    pixel_stepper_pos = (stepper_pos - min_pos) / steps_to_pixels + (int(x_size/2) - int(stepper_stroke/2) /steps_to_pixels) 
    pixel_stepper_pos = pixel_stepper_pos.astype(int)
    logging.debug(pixel_stepper_pos)
    
    sample_file = '{}StepperScan_{}.h5'.format(path,int(stepper_pos[0]))
    y_size = len(analysis.get_lineout(sample_file))
    y_scale = utils.get_attr(sample_file,'pixel_scale') / drift_length
    reconstruction_image = np.zeros((x_size,y_size))
    
    for pos,pixel_location in zip(stepper_pos,pixel_stepper_pos):
        lineout = analysis.get_lineout('{}StepperScan_{}.h5'.format(path,int(pos)))
        for i in range(slit_width_in_pixels):
            reconstruction_image[pixel_location + i - int(slit_width_in_pixels / 2)] = lineout
    if plotting:
        fig,ax = plt.subplots()
        ax.imshow(reconstruction_image.T,aspect='auto',extent = [-x_size*x_scale/2,x_size*x_scale/2,-y_size*y_scale/2,y_size*y_scale/2])
        ax.set_ylabel('x\' [rad]')
        ax.set_xlabel('x [m]')

    return reconstruction_image.T,x_scale,y_scale

def calculate_emittance(image,x_scale,y_scale,plotting=True):
    moments = analysis.get_array_moments(image,x_scale,y_scale)
    stats = analysis.calculate_ellipse(image,x_scale,y_scale)
    logging.debug(stats)
    logging.debug((x_scale,y_scale))

    y_size,x_size = image.shape
    ext = [-x_size*x_scale/2,x_size*x_scale/2,-y_size*y_scale/2,y_size*y_scale/2]
    
    scale = np.array((x_scale,y_scale))
    midpt_px = np.array((x_size/2,y_size/2))
    midpt_scaled = scale*midpt_px

    #scaling for eigenvectors
    ev = stats[2][0]

    if plotting:
        fig,ax = plt.subplots()
        ax.imshow(image,origin='lower',extent=ext)
        ax.set_aspect('auto')

        for j in range(1,4):
            ell = Ellipse(xy=stats[0] - midpt_scaled,\
                          width=stats[1][0]**(0.5)*j*2,\
                          height=stats[1][1]**(0.5)*j*2,\
                          angle = np.rad2deg(-np.arctan(ev[1]/ev[0])))
            ell.set_facecolor('none')
            ell.set_edgecolor('red')
            ell.set_linewidth(1)
            ax.add_artist(ell)
        ax.axhline(0.0)
    logging.info('rms_x: {} mm'.format(stats[1][0]**0.5*1000))
    logging.info('rms_xp: {} mrad'.format(stats[1][1]**0.5*1000))
    logging.info('Slope: {}'.format(np.arctan(ev[1]/ev[0])))
    logging.info('Geometric Emittance: {} mm.mrad'.format(13.2*np.sqrt(np.linalg.det(moments[1]))))
        
