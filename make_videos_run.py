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


path_videos = '/usr/local/data/sejacob/ANOMALY/data/UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test'
train_test = 'Test'

# print "OPENING LOCAL ANOMS"
# with open(os.path.join('data_stored', 'list_cuboids_test_local.pkl'), 'r') as f:
#     list_cuboids_test = pickle.load(f)
#
# print "START PROCESSING LOCAL ANOMS ON TEST"
# list_video_images = df.make_videos_of_anomalies(list_cuboids_test, path_videos, n_frames=5, size_x=11, size_y=11, threshold=0.5)
#
# print "START MAKING FRAMES OF LOCAL ANOMALIES"
# path_results, list_dirs_of_videos = df.make_anomaly_frames(list_video_images, local=True)
#
# print "START MAKING VIDEOS OF LOCAL ANOMALIES"
# df.make_videos_of_frames(path_results, list_dirs_of_videos, local=True)
#
# del(list_cuboids_test)

print "READING GLOBAL ANOMS"
with open(os.path.join('data_stored', 'list_cuboids_test_global.pkl'), 'r') as f:
    list_cuboids_test = pickle.load(f)

print "START PROCESSING GLOBAL ANOMS ON TEST"
list_video_images = df.make_videos_of_anomalies(list_cuboids_test, path_videos, n_frames=5, size_x=11, size_y=11, threshold=1.0)
print "START MAKING FRAMES OF GLOBAL ANOMALIES"
path_results, list_dirs_of_videos = df.make_anomaly_frames(list_video_images, local=False,threshold=1.0)
print "START MAKING VIDEOS OF GLOBAL ANOMALIES"
df.make_videos_of_frames(path_results, list_dirs_of_videos, local=False,threshold=1.0)


print "START PROCESSING GLOBAL ANOMS ON TEST"
list_video_images = df.make_videos_of_anomalies(list_cuboids_test, path_videos, n_frames=5, size_x=11, size_y=11, threshold=1.5)
print "START MAKING FRAMES OF GLOBAL ANOMALIES"
path_results, list_dirs_of_videos = df.make_anomaly_frames(list_video_images, local=False,threshold=1.5)
print "START MAKING VIDEOS OF GLOBAL ANOMALIES"
df.make_videos_of_frames(path_results, list_dirs_of_videos, local=False,threshold=1.5)