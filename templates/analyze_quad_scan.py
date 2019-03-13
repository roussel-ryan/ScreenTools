import matplotlib.pyplot as plt
import numpy as np
import logging

import ScreenTools.file_processing as fp
import ScreenTools.quad_scan as qs 
import ScreenTools.current as curr
import ScreenTools.image_processing as ip
from ScreenTools import utils
from ScreenTools import plotting
from ScreenTools import analysis
from ScreenTools import quad_scan
from ScreenTools import constraint

###############################
#template file as an example of doing a quad scan analysis,copy and change as needed
###############################

def main():
    #parameters for quad scan
    ##########################
    path = ''
    base_filename = ('','.h5')
    drift_length = 0.4572
    quad_length = 0.1
    gamma = 80
    Bp = gamma*0.511 / 300

    quad_current_to_gradient = quad_scan.peach_quad_c2g
    constraint = constraint.MeanChargeConstraint()

    plotting = True
    reverse_polarity = True
    save_points = False

    params = [path,base_filename,drift_length,quad_length,Bp,quad_current_to_gradient,constraint,plotting,reverse_polarity,save_points]
    
    #processing steps
    ##########################
    file_process(path)
    image_process(path)

    
    #quad scan fitting
    ##########################
    for ax in [0,1]:
        do_fit(params,ax)
    

def file_process(path):
    fp.process_folder(path,same_screen=True)

def image_process(path):
    files = utils.get_files(path,'.h5')

    for f in files:
        ip.mask_file(f)
        ip.filter_file(f)

    for f in files:
        ip.threshold_file(f,manual=True)

def do_fit(p,a):
    c = constraint.MeanChargeConstraint()
    f = quad_scan.fit_quad_scan(p[0],p[2],\
                                p[3],p[4],\
                                p[5],constraints=p[6],\
                                plotting=p[7],\
                                save_points=p[9],\
                                reverse_polarity=p[8],\
                                axis=a,\
                                base_filename=p[1])

    emit = np.sqrt(np.linalg.det(f))
    twiss = f / emit
    logging.info('Axis:{} emit:{} beta:{} alpha:{}'.format(a,emit,twiss[0][0],-1*twiss[0][1]))
    return f
