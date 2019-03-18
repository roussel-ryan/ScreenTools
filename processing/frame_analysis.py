import numpy as np
import matplotlib.pyplot as plt
import h5py

from .. import utils

def average_frames(h5file,frames=None,constraints=None,plotting=False,save_name=None):

    if not frames:
        frames = utils.get_frames(h5file,constraints)

    with h5py.File(h5file,'r+') as f:
        first_frame_data = f['/0/img'][:]
        total_data = np.zeros_like(first_frame_data,dtype=np.int64)

        for frame in frames:
            total_data += f['/{}/img'.format(frame)][:]

        avg_data = total_data/len(frames)

        if save_name:
            #if save_name is given create data group backup,img
            grp = f.create_group(save_name)
            grp.create_dataset('raw',data = avg_data)
            grp.create_dataset('img',data = avg_data)
            grp.create_dataset('frames',data = frames)

    if plotting:
        fig,ax = plt.subplots()
        ax.imshow(total_data)
