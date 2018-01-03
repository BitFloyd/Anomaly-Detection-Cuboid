import matplotlib as mpl
mpl.use('Agg')
import numpy as np
from model_pkg import models
import pickle
from functionals_pkg import feature_fns as ff
from functionals_pkg import argparse_fns as af
from data_pkg import data_fns as df
from keras.models import Model
import os
import sys
from sys import argv
import socket
from functionals_pkg.logging import Logger
from sklearn.cluster import KMeans

print "############################################"
print os.getcwd()
print "############################################"

metric = af.getopts(argv)
print metric

if(socket.gethostname()=='puck'):

    print "############################################"
    print "DETECTED RUN ON PUCK"
    print "############################################"

    path_videos = '/usr/local/data/sejacob/ANOMALY/data/UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test'
    path_videos_train = '/usr/local/data/sejacob/ANOMALY/data/UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train'
    train_test = 'Test'
    verbose = 1
    sys.stdout = Logger(key=os.path.join('Results', 'dictionary_test'))


elif('gpu' in socket.gethostname()):
    print "############################################"
    print "DETECTED RUN ON HELIOS: Probably"
    print "############################################"
    verbose = 0
    os.chdir('/scratch/suu-621-aa/ANOMALY/cuboid')
    path_videos = '/scratch/suu-621-aa/ANOMALY/data/UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test'
    path_videos_train = '/scratch/suu-621-aa/ANOMALY/data/UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train'
    train_test = 'Test'
    sys.stdout = Logger(key=os.path.join('Results', 'dictionary_test'))


else:

    print "############################################"
    print "DETECTED RUN ON GUILLIMIN: Probably"
    print "############################################"
    verbose = 0
    os.chdir('/gs/project/suu-621-aa/sejacob/cuboid/')
    path_videos = '/gs/project/suu-621-aa/sejacob/data/UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test'
    path_videos_train = '/gs/project/suu-621-aa/sejacob/data/UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train'
    train_test = 'Test'
    sys.stdout = Logger(key=os.path.join('Results', 'dictionary_test'))


print "LOADING CUBOIDS AND FEATS"

all_cuboids_normed = np.load(os.path.join('data_stored_temporal', 'all_cuboids_normed.npy'))
all_local_feats_normed = np.load(os.path.join('data_stored_temporal', 'all_local_feats_normed.npy'))

mean_data = np.load(os.path.join('data_stored_temporal', 'cuboid_train_mean.npy'))
std_data = np.load(os.path.join('data_stored_temporal', 'cuboid_train_std.npy'))

size_axis = 12
n_frames = 5

# list_cuboids_test, _, _ = df.make_cuboids_of_videos(path_videos, train_test, size_axis, size_axis, n_frames)
# print "GET LOCAL FEATURE THRESHOLDS"
# alpha_local, theta_local, mean_local, cov_inv_local = models.make_thresholds(all_local_feats_normed, 0.3)
# del(all_local_feats_normed)
#
# print "#######################"
# print "LOCAL FEATURES"
# print "#######################"
# print "ALPHA_LOCAL:", alpha_local, " THETA_LOCAL:", theta_local
# print "#######################"
# thresholds_local = [alpha_local, theta_local]
# print "SETTING LOCAL ANOMS"
# list_cuboids_test = ff.set_anomaly_status_local_temporal(list_cuboids_test, thresholds_local, mean_local, cov_inv_local,
#                                                 mean_data, std_data)
# print "SAVING LOCAL ANOMS"
# with open(os.path.join('data_stored_temporal', 'list_cuboids_test_local.pkl'), 'wb') as f:
#     pickle.dump(list_cuboids_test, f)


print "#################################"
print "STARTING GLOBAL ANALYSIS"
print "#################################"

# print "$$$$$$$$$$$$$$$$$$$$$$$"
# print "READING LOCAL TEST ANOMS"
# print "$$$$$$$$$$$$$$$$$$$$$$$"
#
# with open(os.path.join('data_stored_temporal', 'list_cuboids_test_local.pkl'), 'r') as f:
#     list_cuboids_test = pickle.load(f)
#
# del (list_cuboids_test)



print "#################################"
print "SETTING GLOBAL ANOMS"
print "#################################"


print "#################################"
print "GETTING THE MODEL"
print "#################################"

model_name = 'cnn_model_32_hid.h5'
model = models.small_2d_conv_net(size_y=size_axis, size_x=size_axis, n_channels=1, n_frames=n_frames, h_units=32)
model.load_weights(os.path.join('saved_models', model_name))
model.summary()


model = Model(inputs=model.input, outputs=model.layers[19].output)
all_global_feats = model.predict(all_cuboids_normed, verbose=verbose)

np.save(os.path.join('data_stored_temporal','cnn_model_32_hid_predicted_trains.npy'),all_global_feats)

mean_feats = all_global_feats.mean(axis=0)
std_feats = all_global_feats.std(axis=0)

all_global_feats = (all_global_feats - mean_feats)/std_feats

print "$$$$$$$$$$$$$$$$$$$$$$$"
print "FITTING THE KMEANS OBJECT"
print "$$$$$$$$$$$$$$$$$$$$$$$"

kmeans_obj = KMeans(n_clusters=int(metric['-n']), verbose=1,n_jobs=-1)

kmeans_obj.fit(all_global_feats)



print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
print "LOADING THE TRAIN LIST CUBOIDS:"
print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

train = 'Train'
list_cuboids_train, _, _ = df.make_cuboids_of_videos(path_videos_train, train, size_axis, size_axis, n_frames)


print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
print "CREATE DICTIONARY:"
print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
dictionary = ff.make_dictionary(list_cuboids_train, kmeans_obj, model,mean_data, std_data,mean_feats,std_feats)

del(list_cuboids_train)

print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
print "NUMBER OF DICTIONARY ENTRIES:", len(dictionary)
print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

print "Making rows into tuples"
dictionary = [tuple(row) for row in dictionary]
print "Making dictionary into set"
dictionary = list(set(dictionary))
print "Making dictionary back into lists from tuples"
dictionary = [list(row) for row in dictionary]

# print "Making rows into tuples"
# dictionary = [tuple(row) for row in dictionary]
# print "Counting frequencies"
# d = {x:dictionary.count(x) for x in dictionary}
#
# print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
# print "MIN_FREQUENCY_TRAIN:", min(d.values())
# print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
#
# print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
# print "MAX_FREQUENCY_TRAIN:", max(d.values())
# print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
#
# print "Getting keys for dictionary"
# # d = {k: v for k, v in d.iteritems() if v >=1}
# dictionary = d.keys()
# print "Making tuples back into lists"
# dictionary = [list(row) for row in dictionary]

print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
print "NUMBER OF DICTIONARY ENTRIES AFTER FILTERING:", len(dictionary)
print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"


print "$$$$$$$$$$$$$$$$$$$$$$$"
print "READING LOCAL TEST ANOMS"
print "$$$$$$$$$$$$$$$$$$$$$$$"

with open(os.path.join('data_stored_temporal', 'list_cuboids_test_local.pkl'), 'r') as f:
    list_cuboids_test = pickle.load(f)

print ""


list_cuboids_test, list_frequencies = ff.dictionary_based_anom_setting(list_cuboids_test,model,kmeans_obj,dictionary,
                                                                       mean_data, std_data,mean_feats,std_feats)

print "$$$$$$$$$$$$$$$$$$$$$$$"
print "MIN_FREQUENCY:         ",min(list_frequencies)
print "$$$$$$$$$$$$$$$$$$$$$$$"

print "$$$$$$$$$$$$$$$$$$$$$$$"
print "MAX_FREQUENCY:         ",max(list_frequencies)
print "$$$$$$$$$$$$$$$$$$$$$$$"

with open(os.path.join('data_stored_temporal', 'list_frequencies.pkl'), 'wb') as f:
    pickle.dump(list_frequencies,f)

print "SAVING GLOBAL ANOMS"

with open(os.path.join('data_stored_temporal', 'list_cuboids_test_global.pkl'), 'wb') as f:
    pickle.dump(list_cuboids_test, f)


# print "READING GLOBAL ANOMS"
# with open(os.path.join('data_stored_temporal', 'list_cuboids_test_global.pkl'), 'r') as f:
#     list_cuboids_test = pickle.load(f)

print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
print "START PROCESSING GLOBAL ANOMS ON TEST"
print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
list_video_images = df.make_videos_of_anomalies(list_cuboids_test, path_videos, n_frames=n_frames, size_x=size_axis, size_y=size_axis,
                                                threshold=2.0,cnn=True)
print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
print "START MAKING FRAMES OF GLOBAL ANOMALIES"
print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
path_results, list_dirs_of_videos = df.make_anomaly_frames(list_video_images, local=False,threshold=2.0)

print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
print "START MAKING VIDEOS OF GLOBAL ANOMALIES"
print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
df.make_videos_of_frames(path_results, list_dirs_of_videos, local=False,threshold=2.0)
