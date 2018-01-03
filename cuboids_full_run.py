import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from model_pkg import models
import pickle
from functionals_pkg import feature_fns as ff
from data_pkg import data_fns as df
from keras.models import load_model
from keras.models import Model
import os


print "LOADING CUBOIDS AND FEATS"

all_cuboids_normed = np.load(os.path.join('data_stored', 'all_cuboids_normed.npy'))
all_local_feats_normed = np.load(os.path.join('data_stored', 'all_local_feats_normed.npy'))

mean_data = np.load(os.path.join('data_stored', 'cuboid_train_mean.npy'))
std_data = np.load(os.path.join('data_stored', 'cuboid_train_std.npy'))


path_videos = '/usr/local/data/sejacob/ANOMALY/data/UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test'
train_test = 'Test'

#
# list_cuboids_test, _ = df.make_cuboids_of_videos(path_videos, train_test, 11, 11, 5)
#
# print "GET LOCAL FEATURE THRESHOLDS"
# alpha_local, theta_local, mean_local, cov_inv_local = models.make_thresholds(all_local_feats_normed, 0.3)
#
# del(all_local_feats_normed)
#
# print "#######################"
# print "LOCAL FEATURES"
# print "#######################"
# print "ALPHA_LOCAL:", alpha_local, " THETA_LOCAL:", theta_local
# print "#######################"
#
# thresholds_local = [alpha_local, theta_local]
#
#
# print "SETTING LOCAL ANOMS"
#
# list_cuboids_test = ff.set_anomaly_status_local(list_cuboids_test, thresholds_local, mean_local, cov_inv_local,
#                                                 mean_data, std_data)
#
# print "SAVING LOCAL ANOMS"
# with open(os.path.join('data_stored', 'list_cuboids_test_local.pkl'), 'wb') as f:
#     pickle.dump(list_cuboids_test, f)

# print "MAKING VIDEOS OF LOCAL ANOMS"
# list_video_images = df.make_videos_of_anomalies(list_cuboids_test, path_videos, n_frames=5, size_x=11, size_y=11, threshold=0.5)
#
# path_results, list_dirs_of_videos = df.make_anomaly_frames(list_video_images, local=True)
#
# df.make_videos_of_frames(path_results, list_dirs_of_videos, local=True)

print "#################################"
print "STARTING GLOBAL ANALYSIS"
print "#################################"


print "READING LOCAL ANOMS"
with open(os.path.join('data_stored', 'list_cuboids_test_local.pkl'), 'r') as f:
    list_cuboids_test = pickle.load(f)

print "SETTING GLOBAL ANOMS"

print "LOADING MODEL AND GETTING FEATS FROM TRAIN CUBOIDS"
model = load_model(os.path.join('saved_models', 'same_as_paper_sigmoid.h5'))
print 'same_as_paper_sigmoid.h5'

model = Model(inputs=model.input, outputs=model.layers[2].output)
all_global_feats = model.predict(all_cuboids_normed, verbose=1)
print ""

del(all_cuboids_normed)

print "GET GLOBAL FEATURE THRESHOLDS"
alpha_global, theta_global, mean_global, cov_inv_global = models.make_thresholds(all_global_feats, 0.2)

print "#######################"
print "GLOBAL FEATURES"
print "#######################"
print "ALPHA_GLOBAL:", alpha_global, " THETA_GLOBAL:", theta_global
print "#######################"

thresholds_global = [alpha_global, theta_global]

list_cuboids_test = ff.set_anomaly_status_global(list_cuboids_test, thresholds_global, model, mean_global,
                                                 cov_inv_global, mean_data, std_data)

print "SAVING GLOBAL ANOMS"
with open(os.path.join('data_stored', 'list_cuboids_test_global.pkl'), 'wb') as f:
    pickle.dump(list_cuboids_test, f)


# del(list_cuboids_test)
#
# print "READING GLOBAL ANOMS"
# with open(os.path.join('data_stored', 'list_cuboids_test_global.pkl'), 'r') as f:
#     list_cuboids_test = pickle.load(f)

# print "Starting making videos"
# path_videos = '/usr/local/data/sejacob/ANOMALY/data/UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test'
# train_test = 'Test'
#
# list_video_images = df.make_videos_of_anomalies(list_cuboids_test, path_videos, n_frames=5, size_x=11, size_y=11, threshold=1.5)
#
# path_results, list_dirs_of_videos = df.make_anomaly_frames(list_video_images, local=False)
# df.make_videos_of_frames(path_results, list_dirs_of_videos, local=False)


