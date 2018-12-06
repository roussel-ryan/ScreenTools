import numpy as np
import matplotlib.pyplot as plt
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
    #for file in files:
    #    ip.reset_file(file)
    #    ip.mask(file)
    #    ip.filter(file,filter_high_res = True)
    #    ip.remove_background(file)
    for file in files:
        ax = plotting.plot_screen(file)
        plotting.overlay_lineout(file,ax)

def reconstruct_phase_space(path,drift_length,slit_width,steps_to_m):
    files = utils.get_files(path,'.h5')
    logging.debug(files)
    reconstruction_image = []
    
    #get stepper positions
    stepper_pos = []
    for file in files:
        stepper_pos.append(file.split('StepperScan_')[1].split('.h5')[0])
    stepper_pos = np.sort(np.asfarray(stepper_pos))
    
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
    
    sample_file = 'TQ3_222/StepperScan_{}.h5'.format(int(stepper_pos[0]))
    y_size = len(analysis.get_lineout(sample_file))
    y_scale = 50.0e-3 / utils.get_attr(sample_file,'screen_radius')*2.0 / drift_length
    reconstruction_image = np.zeros((x_size,y_size))
    
    for pos,pixel_location in zip(stepper_pos,pixel_stepper_pos):
        lineout = analysis.get_lineout('TQ3_222/StepperScan_{}.h5'.format(int(pos)))
        for i in range(slit_width_in_pixels):
            reconstruction_image[pixel_location + i - int(slit_width_in_pixels / 2)] = lineout
    
    #image = np.asfarray(reconstruction_image).T
    fig,ax = plt.subplots()
    
    ax.imshow(reconstruction_image.T,aspect='auto',extent = [-x_size*x_scale/2,x_size*x_scale/2,-y_size*y_scale/2,y_size*y_scale/2])
    ax.set_ylabel('x\' [rad]')
    ax.set_xlabel('x [m]')
    
    
        
