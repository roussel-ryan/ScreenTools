import matplotlib.pyplot as plt
import numpy as np
import h5py
import logging
import os

from . import utils
from .processing import thresholding

def fwhm(y):
    x = range(len(y))
    half_max = np.max(y) / 2
    a = np.argwhere(y > half_max)
    return x[np.max(a)] - x[np.min(a)]

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
    get_charge(h5filename,overwrite=overwrite)
    return h5filename

def add_ICT_scale(h5filename,scale):
    with h5py.File(h5filename,'r+') as f:
        f['/'].attrs['ICT_scale'] = scale
    

def get_charge(h5filename,frame_number=-1,constraints=None,plotting=False,overwrite=False):
    ''' add charge attribute to group if current dataset 
        is found
    '''
    data = []
    with h5py.File(h5filename) as f:
        if frame_number == -1:
            frame_number = utils.get_frames(h5filename,constraints)
        elif not isinstance(frame_number,list):
            frame_number = [frame_number]
            
        ICT_scale = 2.5*(10 / f.attrs['ICT_scale'])
        for i in frame_number:
            try:
                if overwrite:
                    del f['/{}/'.format(i)].attrs['charge']
                data.append(f['/{}/'.format(i)].attrs['charge'])
            except KeyError:
                try:
                    current = f['/{}/current'.format(i)][:]
                    #logging.debug(current)
                    #logging.debug(np.sum(current[1]))
                    #threshold data
                    t = current[0]
                    current = -current[1:]
                    tcurrent = []

                    #for ICT1 get FWHM size
                    h,be = np.histogram(current[0][-1000:],bins=200)
                    bc = (be[1:] + be[:-1]) / 2
                    m = np.average(bc,weights=h)
                    #tcurrent.append(current[0] - m)

                    ICT1_fwhm = fwhm(current[0] - m)
                    logging.debug(ICT1_fwhm)
                
                    for c in current:
                        if np.std(c) < 1.0e-10:
                            tcurrent.append(0.0*c)
                        else:
                            #mask out the peak region to find the background
                            peak_index = np.argmax(c)
                            logging.debug(peak_index)
                            n = np.arange(len(c))
                            mask = np.where((n > peak_index - ICT1_fwhm*1.5)*(n < peak_index + ICT1_fwhm*3.5),0,1)
                            anti_mask = -(mask-1)
                            h,be = np.histogram(c*mask,bins=200)
                            bc = (be[1:] + be[:-1]) / 2
                            m = np.average(bc,weights=h)
                            tcurrent.append((c - m)*anti_mask)

                    
                
                    charge = [np.trapz(tcurrent[i]/ICT_scale[i],t) for i in range(len(tcurrent))]
                
                    f['/{}'.format(i)].attrs['charge'] = np.asfarray(charge)
                
                    data.append(charge)
                except KeyError:
                    logging.warning('/{}/current not found in {}'.format(i,h5filename))

                if plotting:
                    fig,ax = plt.subplots()
                    for ele in tcurrent:
                        ax.plot(t,ele)

    return np.asfarray(data)
                
def calculate_charge_stats(h5filename,constraints=None):    
    data = get_charge(h5filename,constraints=constraints,overwrite=True).T
    means = []
    stds = []
    for ele in data:
        means.append(np.mean(ele))
        stds.append(np.std(ele))
    with h5py.File(h5filename,'r+') as f:
        f['/'].attrs['avg_charge'] = means
        f['/'].attrs['std_charge'] = stds
    return (means,stds)
    

#for use in utils.get_frames
def mean_charge(f,ID,ICT_channel = 0):
    #returns true if frame current is within +/- 1 sigma of the mean
    return abs(f['/{}'.format(ID)].attrs['charge'][ICT_channel] - f['/'].attrs['avg_charge'][ICT_channel]) < f['/'].attrs['std_charge'][ICT_channel]
        
def selected_charge(f,ID,lower_limit,upper_limit,ICT_channel):
    #returns true if charge @ ICT5 is 2.0 +/- 0.5 nC
    charge = f['/{}'.format(ID)].attrs['charge'][ICT_channel]
    return (charge > lower_limit and charge <= upper_limit) 
    
    
            
