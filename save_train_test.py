from data_pkg import data_fns as df
from functionals_pkg import feature_fns as ff
import numpy as np
import pickle
import os

print "TRAINING : "
path_videos = '/usr/local/data/sejacob/ANOMALY/data/UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train'
train_test = 'Train'

list_cuboids, all_cuboids = df.make_cuboids_of_videos(path_videos, train_test, 11, 11, 5)

mean = all_cuboids.mean(axis=0)
std = all_cuboids.std(axis=0)

all_cuboids = (all_cuboids - mean) / std

np.save(os.path.join('data_stored', 'cuboid_train_mean.npy'), mean)
np.save(os.path.join('data_stored', 'cuboid_train_std.npy'), std)
np.save(os.path.join('data_stored', 'all_cuboids_normed.npy'), all_cuboids)

with open(os.path.join('data_stored', 'list_cuboids.pkl'), 'wb') as f:
    pickle.dump(list_cuboids, f)

len_local_feats = 13
feats_local = ff.make_all_local_feats(len_local_feats, all_cuboids=list_cuboids, mean=mean, std=std)

np.save(os.path.join('data_stored', 'all_local_feats_normed.npy'), feats_local)

del (list_cuboids)
del (all_cuboids)
del (feats_local)
