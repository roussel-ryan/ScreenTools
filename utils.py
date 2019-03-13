import h5py
import numpy as np
import logging
from os import listdir
from os.path import isfile, join

from . import constraint

def get_frames(filename,constraints=None):
    ''' return list of frame numbers which satisfy certain conditions given by a single callable function or an array of callable functions

    functions takes a Constraint object
    '''
    #check constraints
    if constraints == None:
        constraints = [constraint.NullConstraint()]
    elif isinstance(constraints,list):
        for ele in constraints:
            if not isinstance(ele,constraint.Constraint):
                logging.error('Error object {} not constraint!'.format(ele))
    elif isinstance(constraints,constraint.Constraint):
        constraints = [constraints]
    else:
        logging.error('Error object {} not constraint!'.format(constraints))

    #apply constraints
    with h5py.File(filename) as f:
        nframes = f.attrs['nframes']
        good_frames = []
        for i in range(nframes - 1):
            good = True
            for c in constraints:
                #logging.debug(c.name)
                if not c.evaluate(f,i):
                    good = False
                        #if it survives then add to good frames
                #except:
                #    logging.error('There was an issue evaluating frame {} in file {}'.format(i,filename))
                #    good = False
            if good:
                good_frames.append(i)



    if not len(good_frames) == nframes:
        logging.info('Cutting frames in file {}, percentage remaining:{:.2f}'.format(filename,100 * (len(good_frames)/nframes)))
        
    return good_frames

                       
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

def print_attrs(filename,dataset='/'):
    logging.info(get_attrs(filename,dataset))

def get_files(path,file_ext):
    files = [f for f in listdir(path) if isfile(join(path,f))]
    dat_files = [join(path,f) for f in files if file_ext in f]
    return dat_files
          
def print_items(filename):
    def print_item(name, obj):
        logging.info(name)
        for key, val in obj.attrs.items():
            logging.info("    %s: %s" % (key, val))
    logging.info('printing items in filename ' + filename)
    with h5py.File(filename,'r') as f:
        f.visititems(print_item)

def print_image_stats(filename):
    with h5py.File(filename,'r') as f:
        for i in range(f.attrs['nframes']):
            out = '{} '.format(i)
            for name,val in f['/{}'.format(i)].attrs.items():
                out += '{}:{} '.format(name,val)
            logging.info(out)
