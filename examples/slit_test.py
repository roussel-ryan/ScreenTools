import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage

import logging
import h5py

import ScreenTools.slit_scan as ss
import ScreenTools.utils as utils
import ScreenTools.image_processing as ip
import ScreenTools.plotting as plotting
import ScreenTools.analysis as analysis
import ScreenTools.current as current

def check_thresholds():
    path = 'Slit_scan_PT2/'
    files = utils.get_files(path,'.h5')
    for fname in files:
        with h5py.File(fname) as f:
            logging.debug(f['/0/'].attrs['threshold'])
    ss.plot_files(path,overlay=False)
            
def main():
    path = 'Slit_scan_PT2/'
    test_file = '{}StepperScan_-233638.h5'.format(path)
    #files = ss.process_files(path)
    #ss.plot_files(path)
    files = utils.get_files(path,'.h5')
    #logging.debug(files)
    utils.print_attrs(test_file)
    #ss.process_files(path)
    #ss.threshold_files(path)

    fig,ax = plt.subplots()

    #ax.plot(analysis.get_averaged_projection(test_file))
    #plotting.plot_screen(test_file)
    #locs = ss.map_slit_locations(path,100.0e-6,1.5875e-7,plotting=True)
    #ss.map_projections(path,locs,100.0e-6,1.0,condition=None,plotting=True)
    
    #for ele in files:
        #ip.reset_file(ele)
        #ip.mask_file(ele)
        #ip.threshold_file(ele)
        #ip.plot_threshold(ele)
        #plotting.plot_screen(ele)
        #plotting.plot_screen(ele,'/raw')
        #current.add_charge(ele)
        
    ps = ss.reconstruct_phase_space(path,1.0,100.0e-6,1.5875e-7,plotting=True)
    ss.calculate_emittance(*ps)
    #ss.calculate_emittance(ps[0],1,1)
logging.basicConfig(level=logging.DEBUG)
main()
#check_thresholds()
plt.show()
