
#process files in folder

from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
import logging
import subprocess
import h5py

from . import utils
from . import screen_finder

def process_folder(path,same_screen=False,**kwargs):
    '''
    go through and process all data files in "path"
    - removes blackfly from file names
    - converts files from dat format to h5 format
    - prompts user to select points on screen image to get screen coordinates
    
    OPTIONS:
    --------
    same_screen = False             specify if ALL files are on the same screen
                                    will only screen find the first file and add
                                    same screen attrs to each file in folder
    overwrite = False               re-find the screen on found files in "path"
    skip_screen_finder = False      skip screen finder on found files in "path"
    '''

    dat_files = get_files(path)
    logging.debug(dat_files)
    #go through and remove blackfly crap
    for f in dat_files:
        if '_Blackfly' in f:
            base_name = f.split('_Blackfly')[0]
            logging.debug(base_name)
            subprocess.run(['mv',f,'{}.dat'.format(base_name)])
            
    dat_files = get_files(path)
    #cycle through dat_files in path    
    h5_files = []
    if same_screen:
        logging.info('running {} processing in scame_screen mode,warings suppressed for extra files'.format(path))
        h5_files.append(process_raw(dat_files[0],skip_screen_finder = False,\
                                    overwrite=True,**kwargs))
        with h5py.File(h5_files[0]) as f:
            screen_center = f.attrs['screen_center']
            screen_radius = f.attrs['screen_radius']
        
        
        for filename in dat_files[1:]:
            h5_files.append(process_raw(filename,skip_screen_finder=True,\
                                        overwrite=True,suppress_warning = True,**kwargs))
            with h5py.File(h5_files[-1]) as f:
                f.attrs['screen_center'] = screen_center
                f.attrs['screen_radius'] = screen_radius
                f.attrs['pixel_scale'] = 50.038e-3 / (screen_radius*2)
    else:
        for filename in dat_files:
            h5_files.append(process_raw(filename,**kwargs))
    

    
    return h5_files

def add_current(h5filename):
    
    fbase = h5filename.split('_img')[0].split('.')[0]
    if isfile(fbase + 'LeCroy.sdds'):
        with open(fbase + 'LeCroy.sdds') as f:
            datasets = []
            for line in f:
                pt = line.strip().split('\t')
                if len(pt) == 2:
                    if 'time' in line:
                        datasets.append([])
                    else:
                        datasets[-1].append(pt)
                
        #logging.info(datasets[1])
        datasets = np.asfarray(datasets)
        
    elif isfile(fbase + '_LeCroy.csv'):
        with open(fbase + '_LeCroy.csv') as f:
            datasets = []
            for line in f:
                pt = line.strip().replace(' ','').split(',')
                if len(pt) >= 2 and not 'time' in pt[0]:
                    datasets[-1].append(pt)
                else:
                    if 'Number' in line: datasets.append([])
        #logging.info(datasets[1])
        datasets = np.asfarray(datasets)

    else:
        logging.warning('No current file found associated with ' + h5filename)
    
    with h5py.File(h5filename,'r+') as f:
        try:
            for i,dset in zip(range(len(datasets)),datasets):
                f.create_dataset('/{}/current'.format(i),data=dset.T)
        except RuntimeError:
            logging.error('File {} already has current data!'.format(h5filename))

def convert_to_h5(filename,header_size=6,order_type='C'):
    ''' import image data into h5 format for more efficient computing'''
    logging.info('reading data from file: {}'.format(filename))
    data    = np.fromfile(filename, dtype=np.uint16)
    dx = int(data[1])
    dy = int(data[0])

    logging.debug((dx,dy,data[:7]))
    
    if header_size==6:
        nframes = int(data[2])+1
        length  = dx*dy*nframes   
        n = header_size# + 1 
        images  = data[n:]
    else:
        nframes = int(data[2])
        length  = dx*dy*nframes   
        n = header_size + 1 
        images  = data[n:]

    logging.debug(images.shape)
        
    nimages = np.reshape(images,(-1, dx, dy), order=order_type)
    #logging.debug(nimages.shape)
    
    #now send all the images to preallocated datasets
    with h5py.File('{}.h5'.format(filename.split('.')[0]),'w') as f:
        #attach metadata to main group
        f.attrs['dx'] = dx
        f.attrs['dy'] = dy
        f.attrs['nframes'] = nframes
        
        for i in range(len(nimages)):
            grp = f.create_group('{}'.format(i))

            grp.create_dataset('img'.format(i),data=nimages[i],dtype=np.uint16)
            grp.create_dataset('raw'.format(i),data=nimages[i],dtype=np.uint16)
    
    logging.info('Done importing')    
    return None 

def process_raw(filename,overwrite=False,skip_screen_finder=False,suppress_warning = False,header_size = 6):
    if not filename.split('.')[1] == 'dat':
        logging.error('filename {} not .dat file!'.format(filename))
        return None
    
    h5_filename = '{}.h5'.format(filename.split('.')[0])
    if not isfile(h5_filename) or overwrite:
        if overwrite and isfile(h5_filename) and not suppress_warning:
            logging.warning('WARNING: About to overwrite h5 file {}, proceed with caution!!!! Press enter to confirm.'.format(h5_filename))
            input('----')
        convert_to_h5(filename,header_size=header_size)
        add_current(h5_filename)
    logging.debug('searching for file ' + h5_filename)
    
    if not skip_screen_finder:
        if overwrite:       
            screen_finder.ScreenFinder(h5_filename)
        else:
            attrs = list(utils.get_attrs(h5_filename).keys())
            if not ('screen_center' in attrs and 'screen_radius' in attrs):
                screen_finder.ScreenFinder(h5_filename)
    

    return h5_filename
                
def get_files(path):
    files = [f for f in listdir(path) if isfile(join(path,f))]
    dat_files = [join(path,f) for f in files if '.dat' in f]
    return dat_files
    
if __name__=='__main__':
    logging.basicConfig(level=logging.DEBUG)
    process('10_17/',rewrite_stat=False)
