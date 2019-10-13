import numpy as np

import scipy.io as sio

def process_data(args):
    # to do release other dataset.
    if 'rgbd' in args.path:
        data = sio.loadmat('./Data/rgbd_mtv.mat')
        features = data['X']
        label = data['gt']

        views = []
        view_shape = []
        for v in features[0]:
            view_shape.append(v.shape[1])
            views.append(v)
        
        return view_shape, views, label