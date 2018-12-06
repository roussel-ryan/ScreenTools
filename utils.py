import h5py
import numpy as np
import logging
from os import listdir
from os.path import isfile, join

def get_attr(filename,attr,dataset='/'):
    with h5py.File(filename,'r') as f:
        try:
            
            var = f[dataset].attrs[attr]
            return var
        except KeyError:
            return None
          
def get_attrs(filename,dataset='/'):
    r = {}
    with h5py.File(filename,'r') as f:
        for item,val in f.attrs.items():
            r[item] = val
    return r
          
def get_files(path,file_ext):
    files = [f for f in listdir(path) if isfile(join(path,f))]
    dat_files = [join(path,f) for f in files if file_ext in f]
    return dat_files
          
def print_items(filename):
    def print_attrs(name, obj):
        logging.info(name)
        for key, val in obj.attrs.items():
            logging.info("    %s: %s" % (key, val))
    logging.info('printing items in filename ' + filename)
    with h5py.File(filename,'r') as f:
        f.visititems(print_attrs)

def print_image_stats(filename):
    with h5py.File(filename,'r') as f:
        for i in range(f.attrs['nframes']):
            out = '{} '.format(i)
            for name,val in f['/{}'.format(i)].attrs.items():
                out += '{}:{} '.format(name,val)
            logging.info(out)
