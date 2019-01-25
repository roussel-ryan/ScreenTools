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

def main():
    #process files
    path = 'PYG1_QUAD_SCAN/'
    files = utils.get_files(path,'.h5')
    #fp.process_folder(path,same_screen=True)
    #ip.filter_file(files[0])
    #ip.threshold_file(files[0],overwrite=True,plotting=True)
    
    for f in files:
        ip.mask_file(f)
        ip.filter_file(f)
        ip.threshold_file(f)

    test_filea = '{}QUADSCAN_-9_0_img.h5'.format(path)
    test_fileb = '{}QUADSCAN_4_5_img.h5'.format(path)

    plotting.plot_screen(test_filea,scaled=True)
    plotting.plot_screen(test_fileb,scaled=True)
    
    #plotting.plot_screen('{}QUADSCAN_-9_0_img.h5'.format(path),dataset='/raw')
    #logging.debug(analysis.get_moments(test_file,overwrite=True))
    
    qs.fit_quad_scan(path,0.35,0.1,0.13,qs.radiabeam_quad_current_to_gradient,conditions=curr.mean_charge,plotting=True)
    #qs.fit_quad_scan(path,1,1,conditions=curr.mean_charge,plotting=True)
    
if __name__=='__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
    plt.show()
