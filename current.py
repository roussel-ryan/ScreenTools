import numpy as np
import h5py
import logging
import os

from . import utils

def add_current(h5filename):
    '''
    adds current distribution from LeCroy to each image group

    Inputs:
    -------
    h5filename      h5 filename which contains image data

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
    with h5py.File(h5filename,'r+') as f:
        try:
            for i,dset in zip(range(len(datasets)),datasets):
                f.create_dataset('/{}/current'.format(i),data=dset.T)
        except RuntimeError:
            logging.error('File {} already has current data!'.format(h5filename))
    return h5filename

def add_charge(h5filename):
    ''' add charge attribute to group if current dataset 
        is found
    '''
    data = []
    with h5py.File(h5filename) as f:
        nframes = f.attrs['nframes']
        for i in range(nframes - 1):
            try:
                current = f['/{}/current'.format(i)][:]
                #logging.debug(current)
                #logging.debug(np.sum(current[1]))
                f['/{}'.format(i)].attrs['charge'] = -np.sum(current[1])
                data.append(-np.sum(current[1]))
            except KeyError:
                logging.warning('/{}/current not found in {}'.format(i,h5filename))

        data = np.asfarray(data)
        f.attrs['avg_charge'] = np.mean(data) 
        f.attrs['std_charge'] = np.std(data)

#for use in utils.get_frames
def mean_charge(f,ID):
    #returns true if frame current is within +/- 1 sigma of the mean
    return abs(f['/{}'.format(ID)].attrs['charge'] - f['/'].attrs['avg_charge']) < f['/'].attrs['std_charge']
        

    
    
            
