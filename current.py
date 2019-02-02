import matplotlib.pyplot as plt
import numpy as np
import h5py
import logging
import os

from . import utils
from .processing import thresholding

def add_current(h5filename,ICT_scale=[20,10,10,10],overwrite=False):
    '''
    adds current distribution from LeCroy to each image group

    Inputs:
    -------
    h5filename      h5 filename which contains image data
    ICT_scale       sensitivity ratio for each channel (either 20:1 or 10:1)
                    for 20:1 multiply trace by 1.25, for 10:1 multiply by 2.5

    '''
    
    fbase = h5filename.split('_img')[0].split('.')[0]
    if os.path.isfile(fbase + 'LeCroy.sdds'):
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
        
    elif os.path.isfile(fbase + '_LeCroy.csv'):
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
        raise RuntimeError
    add_ICT_scale(h5filename,scale=ICT_scale)
    
    with h5py.File(h5filename,'r+') as f:
        try:
            for i,dset in zip(range(len(datasets)),datasets):
                if overwrite:
                    del f['/{}/current'.format(i)]
                
                f.create_dataset('/{}/current'.format(i),data=dset.T)
        except RuntimeError:
            logging.error('File {} already has current data!'.format(h5filename))
    return h5filename

def add_ICT_scale(h5filename,scale):
    with h5py.File(h5filename,'r+') as f:
        f['/'].attrs['ICT_scale'] = scale
    

def add_charge(h5filename,plotting=False):
    ''' add charge attribute to group if current dataset 
        is found
    '''
    data = []
    with h5py.File(h5filename) as f:
        nframes = f.attrs['nframes']
        ICT_scale = 2.5*(10 / f.attrs['ICT_scale'])
        for i in range(nframes):
            try:
                current = f['/{}/current'.format(i)][:]
                #logging.debug(current)
                #logging.debug(np.sum(current[1]))
                #threshold data
                t = current[0]
                current = -current[1:]
                tcurrent = []
                for c in current:
                    threshold = thresholding.calculate_threshold(c)
                    tcurrent.append(np.where(c > threshold,c,0))
                
                charge = [np.trapz(tcurrent[i]/ICT_scale[i],t) for i in range(len(tcurrent))]
                
                f['/{}'.format(i)].attrs['charge'] = np.asfarray(charge)
                
                data.append(charge)
            except KeyError:
                logging.warning('/{}/current not found in {}'.format(i,h5filename))

        if plotting:
            fig,ax = plt.subplots()
            for ele in tcurrent:
                ax.plot(t,ele)
                
        data = np.asfarray(data)
        f.attrs['avg_charge'] = np.mean(data,axis=0) 
        f.attrs['std_charge'] = np.std(data,axis=0)

def get_frame_charge(filename,frame_number=0,overwrite=False,plotting=False):
    #avg_charge = utils.get_attr(filename,'avg_charge')
    #if avg_charge or not overwrite:
    #    charge = utils.get_attr(filename,'charge','/{}'.format(frame_number))
    #else:
    add_charge(filename,plotting)
    charge = utils.get_attr(filename,'charge','/{}'.format(frame_number))
    return charge

def get_file_charge(filename,overwrite=False):
    avg_charge = utils.get_attr(filename,'avg_charge')
    std_charge = utils.get_attr(filename,'std_charge')
    return (avg_charge,std_charge)



#for use in utils.get_frames
def mean_charge(f,ID):
    #returns true if frame current is within +/- 1 sigma of the mean
    return abs(f['/{}'.format(ID)].attrs['charge'] - f['/'].attrs['avg_charge']) < f['/'].attrs['std_charge']
        
def selected_charge(f,ID,lower_limit,upper_limit,ICT_channel):
    #returns true if charge @ ICT5 is 2.0 +/- 0.5 nC
    charge = f['/{}'.format(ID)].attrs['charge'][ICT_channel]
    return (charge > lower_limit and charge <= upper_limit) 
    
    
            
