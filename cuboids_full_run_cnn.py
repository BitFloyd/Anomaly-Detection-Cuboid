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
from keras.layers import Flatten
import os
from scipy.stats import multivariate_normal


print "LOADING CUBOIDS AND FEATS"

all_cuboids_normed = np.load(os.path.join('data_stored_temporal', 'all_cuboids_normed.npy'))
all_local_feats_normed = np.load(os.path.join('data_stored_temporal', 'all_local_feats_normed.npy'))

mean_data = np.load(os.path.join('data_stored_temporal', 'cuboid_train_mean.npy'))
std_data = np.load(os.path.join('data_stored_temporal', 'cuboid_train_std.npy'))


path_videos = '/usr/local/data/sejacob/ANOMALY/data/UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test'
train_test = 'Test'

size_axis = 12
n_frames = 5

list_cuboids_test, _, _ = df.make_cuboids_of_videos(path_videos, train_test, size_axis, size_axis, n_frames)
print "GET LOCAL FEATURE THRESHOLDS"
alpha_local, theta_local, mean_local, cov_inv_local = models.make_thresholds(all_local_feats_normed, 0.3)
del(all_local_feats_normed)

print "#######################"
print "LOCAL FEATURES"
print "#######################"
print "ALPHA_LOCAL:", alpha_local, " THETA_LOCAL:", theta_local
print "#######################"
thresholds_local = [alpha_local, theta_local]
print "SETTING LOCAL ANOMS"
list_cuboids_test = ff.set_anomaly_status_local_temporal(list_cuboids_test, thresholds_local, mean_local, cov_inv_local,
                                                mean_data, std_data)
print "SAVING LOCAL ANOMS"
with open(os.path.join('data_stored_temporal', 'list_cuboids_test_local.pkl'), 'wb') as f:
    pickle.dump(list_cuboids_test, f)

# print "LOADING LOCAL ANOMS"
# with open(os.path.join('data_stored_temporal', 'list_cuboids_test_local.pkl'), 'r') as f:
#     list_cuboids_test = pickle.load(f)

# print "MAKING VIDEOS OF LOCAL ANOMS"
# list_video_images = df.make_videos_of_anomalies(list_cuboids_test, path_videos, n_frames=n_frames, size_x=size_axis, size_y=size_axis, threshold=0.5,cnn=True)
# path_results, list_dirs_of_videos = df.make_anomaly_frames(list_video_images, local=True)
# df.make_videos_of_frames(path_results, list_dirs_of_videos, local=True)

print "#################################"
print "STARTING GLOBAL ANALYSIS"
print "#################################"


print "READING LOCAL ANOMS"
with open(os.path.join('data_stored_temporal', 'list_cuboids_test_local.pkl'), 'r') as f:
    list_cuboids_test = pickle.load(f)

print "SETTING GLOBAL ANOMS"

size_axis = 12
n_frames = 5

print "GETTING THE MODEL"

model_name = 'cnn_model_128_f_nobn.h5'
model = models.small_2d_conv_net(size_y=size_axis, size_x=size_axis, n_channels=1, n_frames=n_frames)
model.load_weights(os.path.join('saved_models', model_name))
model.summary()



model1 = Model(inputs=model.input, outputs=Flatten()(model.layers[15].output))
model2 = Model(inputs=model.input, outputs=Flatten()(model.layers[11].output))

all_global_feats = np.hstack((model1.predict(all_cuboids_normed, verbose=1),model2.predict(all_cuboids_normed,verbose=1)))

del(model)
print ""

# print "GET GLOBAL FEATURE THRESHOLDS"
# alpha_global, theta_global, mean_global, cov_inv_global = models.make_thresholds(all_global_feats, 0.05)
#
# print "#######################"
# print "GLOBAL FEATURES"
# print "#######################"
# print "ALPHA_GLOBAL:", alpha_global, " THETA_GLOBAL:", theta_global
# print "#######################"
#


# LOG PDF TRAIN AND TEST THRESHOLDS - LEAST 1%
# TRAIN 1%:  -10453.1960461
# TEST 1%:  -14182.3255045
# MAHALANOBIS DIST TRAIN AND TEST THRESHOLDS - TOP 1%
# TRAIN 1%:  171.671245197
# TEST 1%:  192.169912693



print "FIT MV-GAUSSIAN"
mean_mvg = np.mean(all_global_feats, axis=0)
cov_mvg = np.cov(all_global_feats, rowvar=False)
cov_mv_inv = np.linalg.inv(cov_mvg)

var = multivariate_normal(mean=mean_mvg,cov=cov_mvg)

threshold_logpdf = -14182.3255045
threshold_mahad = 192.169912693

# list_cuboids_test = ff.set_anomaly_status_global_mvg(list_cuboids_test, threshold, model, var, mean_data, std_data)
# list_cuboids_test = ff.set_anomaly_status_global_single_th(list_cuboids_test, threshold,model, mean_global,
#                                  cov_inv_global, mean_data, std_data)

list_cuboids_test = ff.set_anomaly_status_global_mahad_logpdf(list_cuboids_test, threshold_logpdf, threshold_mahad, model1,model2,
                                                              var, mean_data, std_data, mean_mvg.reshape((1,mean_mvg.shape[0])),
                                                              cov_mv_inv)

print "SAVING GLOBAL ANOMS"
with open(os.path.join('data_stored_temporal', 'list_cuboids_test_global.pkl'), 'wb') as f:
    pickle.dump(list_cuboids_test, f)


# print "READING GLOBAL ANOMS"
# with open(os.path.join('data_stored_temporal', 'list_cuboids_test_global.pkl'), 'r') as f:
#     list_cuboids_test = pickle.load(f)

print "START PROCESSING GLOBAL ANOMS ON TEST"
list_video_images = df.make_videos_of_anomalies(list_cuboids_test, path_videos, n_frames=n_frames, size_x=size_axis, size_y=size_axis, threshold=2.0,cnn=True)
print "START MAKING FRAMES OF GLOBAL ANOMALIES"
path_results, list_dirs_of_videos = df.make_anomaly_frames(list_video_images, local=False,threshold=2.0)
print "START MAKING VIDEOS OF GLOBAL ANOMALIES"
df.make_videos_of_frames(path_results, list_dirs_of_videos, local=False,threshold=2.0)
