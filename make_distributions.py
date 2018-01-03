import matplotlib as mpl
mpl.use('Agg')
import numpy as np
from functionals_pkg import feature_fns as ff
from functionals_pkg import argparse_fns as af
from model_pkg import models
from keras.models import Model
import os
from sys import argv
import socket
import sys
from functionals_pkg.logging import Logger
from sklearn.decomposition import PCA
import pickle

metric = af.getopts(argv)
print metric

print "############################################"
print os.getcwd()
print "############################################"

if(socket.gethostname()=='puck'):

    print "############################################"
    print "DETECTED RUN ON PUCK"
    print "############################################"

    path_videos = '/usr/local/data/sejacob/ANOMALY/data/UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test'
    train_test = 'Test'
    verbose = 1
    if(metric['-m']=='gaussianmixture' or metric['-m']=='kmeans'):
        sys.stdout = Logger(key=os.path.join('Results', metric['-m']+metric['-gcomps']))

    else:
        sys.stdout = Logger(key=os.path.join('Results', metric['-m']))

elif('gpu' in socket.gethostname()):
    print "############################################"
    print "DETECTED RUN ON HELIOS: Probably"
    print "############################################"
    verbose = 0
    os.chdir('/scratch/suu-621-aa/ANOMALY/cuboid')
    path_videos = '/scratch/suu-621-aa/ANOMALY/data/UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test'
    train_test = 'Test'
    sys.stdout=Logger(key = os.path.join('Results',metric['-m']))


else:

    print "############################################"
    print "DETECTED RUN ON GUILLIMIN: Probably"
    print "############################################"
    verbose = 0
    os.chdir('/gs/project/suu-621-aa/sejacob/cuboid/')
    path_videos = '/gs/project/suu-621-aa/sejacob/data/UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test'
    train_test = 'Test'
    if(metric['-m']=='gaussianmixture'):
        sys.stdout = Logger(key=os.path.join('Results', metric['-m']+metric['-gcomps']))

    else:
        sys.stdout = Logger(key=os.path.join('Results', metric['-m']))

print "LOADING CUBOIDS AND FEATS"
all_cuboids_normed = np.load(os.path.join('data_stored_temporal', 'all_cuboids_normed.npy'))

mean_data = np.load(os.path.join('data_stored_temporal', 'cuboid_train_mean.npy'))
std_data = np.load(os.path.join('data_stored_temporal', 'cuboid_train_std.npy'))

size_axis = 12
n_frames = 5

print "GETTING THE MODEL"
model_name = 'cnn_model_32_hid.h5'
model = models.small_2d_conv_net(size_y=size_axis, size_x=size_axis, n_channels=1, n_frames=n_frames, h_units=32)
model.load_weights(os.path.join('saved_models', model_name))
model.summary()


model = Model(inputs=model.input, outputs=model.layers[19].output)
all_global_feats = model.predict(all_cuboids_normed, verbose=verbose)
np.save(os.path.join('data_stored_temporal','cnn_model_32_hid_predicted_trains.npy'),all_global_feats)

# all_global_feats = np.load(os.path.join('data_stored_temporal','cnn_model_32_hid_predicted_trains.npy'))
print ""



pca = False
pca_obj = None

if (metric['-p']=='True'):
    print "############################################"
    print "PCA of train feats:"
    print "############################################"
    pca = True
    pca_obj = PCA(n_components=int(metric['-c']))
    all_global_feats = pca_obj.fit_transform(all_global_feats)
    np.save(os.path.join('Results','all_global_feats_pcaed-100.npy'),all_global_feats)
    with open(os.path.join('Results','pca_obj_100.pkl'), 'wb') as fid:
        pickle.dump(pca_obj, fid)

normalize = False
min_feats = None
ptp_feats = None

if (metric['-n']=='True'):
    normalize = True
    print "############################################"
    print "Normalizing feats:"
    print "############################################"
    min_feats = all_global_feats.mean(0)
    ptp_feats = all_global_feats.std(0)

    all_global_feats = (all_global_feats - min_feats)/ptp_feats

# model1 = Model(inputs=model.input, outputs=Flatten()(model.layers[15].output))
# model2 = Model(inputs=model.input, outputs=Flatten()(model.layers[11].output))
# del(model)

# all_global_feats = np.hstack((model1.predict(all_cuboids_normed, verbose=1),model2.predict(all_cuboids_normed,verbose=1)))

print "SHAPE OF FEATS:" , all_global_feats.shape


del(all_cuboids_normed)


if(metric['-m']=='logpdf'):
    print "############################################"
    print "STARTING LOGPDF METRIC EVALUATION"
    print "############################################"

    status=False
    status = ff.logpdf_mertrics(path_videos, train_test, size_axis, n_frames, model, all_global_feats, mean_data,std_data,
                                verbose,normalize,min_feats,ptp_feats,pca,pca_obj)

    print "############################################"
    if(status):
        print "SUCCESS"
    else:
        print "FAIL"
    print "############################################"

elif(metric['-m']=='mahad'):
    print "############################################"
    print "STARTING MAHAD METRIC EVALUATION"
    print "############################################"

    status = False
    status = ff.mahad_metrics(path_videos, train_test, size_axis, n_frames, model, all_global_feats, mean_data,std_data,
                              verbose,normalize,min_feats,ptp_feats,pca,pca_obj)

    print "############################################"
    if(status):
        print "SUCCESS"
    else:
        print "FAIL"
    print "############################################"

elif(metric['-m']=='gaussianmixture'):

    print "############################################"
    print "STARTING GAUSSIANMIXTURE METRIC EVALUATION"
    print "############################################"

    status = False
    status = ff.gaussianmixture_metrics(path_videos, train_test, size_axis, n_frames, model, all_global_feats, mean_data,std_data,
                                        verbose,normalize,min_feats,ptp_feats,pca,pca_obj,int(metric['-gcomps']))

    print "############################################"
    if(status):
        print "SUCCESS"
    else:
        print "FAIL"
    print "############################################"

elif(metric['-m']=='kmeans'):

    print "############################################"
    print "STARTING KMEANS METRIC EVALUATION"
    print "############################################"

    status = False
    status = ff.kmeans_clustering(path_videos, train_test, size_axis, n_frames, model, all_global_feats, mean_data,std_data,
                                        verbose,normalize,min_feats,ptp_feats,pca,pca_obj,int(metric['-gcomps']))

    print "############################################"
    if(status):
        print "SUCCESS"
    else:
        print "FAIL"
    print "############################################"

else:
    mean_mvg = np.mean(all_global_feats, axis=0)
    cov_mvg = np.cov(all_global_feats, rowvar=False)
    cov_mv_inv = np.linalg.inv(cov_mvg)

    if(metric ['-m'] =='nn'):
        print "############################################"
        print "STARTING NN METRIC EVALUATION"
        print "############################################"

        status = False
        status = ff.nn_metrics(path_videos, train_test, size_axis, n_frames, model, all_global_feats, mean_data,std_data,cov_mv_inv,
                               verbose,normalize,min_feats,ptp_feats,pca,pca_obj)

        print "############################################"
        if (status):
            print "SUCCESS"
        else:
            print "FAIL"
        print "############################################"

    elif(metric ['-m'] =='kdensity'):

        print "############################################"
        print "STARTING NN METRIC EVALUATION"
        print "############################################"

        status = False
        status = ff.kdensity_metrics(path_videos, train_test, size_axis, n_frames, model, all_global_feats, mean_data,std_data,cov_mv_inv,
                                     verbose,normalize,min_feats,ptp_feats,pca,pca_obj)

        print "############################################"
        if (status):
            print "SUCCESS"
        else:
            print "FAIL"
        print "############################################"
    else:
        print "############################################"
        print "INVALID METRIC SPECIFIED......... TRY AGAIN"
        print "############################################"







