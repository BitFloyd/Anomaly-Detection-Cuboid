import numpy as np
import ssim_fns as sf
from tqdm import tqdm
from model_pkg import models
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from data_pkg import data_fns as df
import os
from model_pkg import models as mdls
from scipy.stats import multivariate_normal
from sklearn.externals import joblib


def make_all_local_feats(len_local_feats, all_cuboids, mean, std, past=False):
    all_local_feats = np.zeros((1, len_local_feats))

    for idx, i in tqdm(enumerate(all_cuboids)):
        local_feats = make_local_feats_from_cuboids(i, mean, std, past)

        all_local_feats = np.vstack((all_local_feats, local_feats))

    all_local_feats = all_local_feats[1:]

    return all_local_feats


def make_local_feats_from_cuboids(cuboids_of_video, mean, std, past=False):
    rows = cuboids_of_video[0].shape[0]
    cols = cuboids_of_video[0].shape[1]

    loc_feats = []

    for i in tqdm(range(1, len(cuboids_of_video))):
        for j in range(1, rows - 1):
            for k in range(1, cols - 1):

                surroundings = []

                if (past):
                    surr_idx = i - 1
                else:
                    surr_idx = i

                surroundings.append(cuboids_of_video[surr_idx][j - 1, k - 1].normed_data(data_mean=mean, data_std=std))
                surroundings.append(cuboids_of_video[surr_idx][j - 1, k].normed_data(data_mean=mean, data_std=std))
                surroundings.append(cuboids_of_video[surr_idx][j - 1, k + 1].normed_data(data_mean=mean, data_std=std))
                surroundings.append(cuboids_of_video[surr_idx][j, k - 1].normed_data(data_mean=mean, data_std=std))
                surroundings.append(cuboids_of_video[surr_idx][j, k + 1].normed_data(data_mean=mean, data_std=std))
                surroundings.append(cuboids_of_video[surr_idx][j + 1, k - 1].normed_data(data_mean=mean, data_std=std))
                surroundings.append(cuboids_of_video[surr_idx][j + 1, k].normed_data(data_mean=mean, data_std=std))
                surroundings.append(cuboids_of_video[surr_idx][j + 1, k + 1].normed_data(data_mean=mean, data_std=std))

                current_cuboid = cuboids_of_video[i][j, k].normed_data(data_mean=mean, data_std=std)

                past_cuboid = cuboids_of_video[i - 1][j, k].normed_data(data_mean=mean, data_std=std)

                loc_feats.append(sf.gen_local_feats(centre_cuboid=current_cuboid, surroundings=surroundings,
                                                    past_cuboid=past_cuboid))

    return np.array(loc_feats)


def make_all_local_feats_temporal(len_local_feats, all_cuboids, mean, std):
    all_local_feats = np.zeros((1, len_local_feats))

    for idx, i in tqdm(enumerate(all_cuboids)):
        local_feats = make_local_feats_from_cuboids_temporal(i, mean, std)

        all_local_feats = np.vstack((all_local_feats, local_feats))

    all_local_feats = all_local_feats[1:]

    return all_local_feats


def make_local_feats_from_cuboids_temporal(cuboids_of_video, mean, std):
    rows = cuboids_of_video[0].shape[0]
    cols = cuboids_of_video[0].shape[1]

    loc_feats = []

    for i in tqdm(range(1, len(cuboids_of_video) - 1)):
        for j in range(1, rows - 1):
            for k in range(1, cols - 1):
                surroundings = []
                surr_idx = i - 1
                surroundings.append(cuboids_of_video[surr_idx][j - 1, k - 1].normed_data(data_mean=mean, data_std=std))
                surroundings.append(cuboids_of_video[surr_idx][j - 1, k].normed_data(data_mean=mean, data_std=std))
                surroundings.append(cuboids_of_video[surr_idx][j - 1, k + 1].normed_data(data_mean=mean, data_std=std))
                surroundings.append(cuboids_of_video[surr_idx][j, k - 1].normed_data(data_mean=mean, data_std=std))
                surroundings.append(cuboids_of_video[surr_idx][j, k + 1].normed_data(data_mean=mean, data_std=std))
                surroundings.append(cuboids_of_video[surr_idx][j + 1, k - 1].normed_data(data_mean=mean, data_std=std))
                surroundings.append(cuboids_of_video[surr_idx][j + 1, k].normed_data(data_mean=mean, data_std=std))
                surroundings.append(cuboids_of_video[surr_idx][j + 1, k + 1].normed_data(data_mean=mean, data_std=std))
                surroundings.append(cuboids_of_video[surr_idx][j, k].normed_data(data_mean=mean, data_std=std))

                surr_idx = i + 1
                surroundings.append(cuboids_of_video[surr_idx][j - 1, k - 1].normed_data(data_mean=mean, data_std=std))
                surroundings.append(cuboids_of_video[surr_idx][j - 1, k].normed_data(data_mean=mean, data_std=std))
                surroundings.append(cuboids_of_video[surr_idx][j - 1, k + 1].normed_data(data_mean=mean, data_std=std))
                surroundings.append(cuboids_of_video[surr_idx][j, k - 1].normed_data(data_mean=mean, data_std=std))
                surroundings.append(cuboids_of_video[surr_idx][j, k + 1].normed_data(data_mean=mean, data_std=std))
                surroundings.append(cuboids_of_video[surr_idx][j + 1, k - 1].normed_data(data_mean=mean, data_std=std))
                surroundings.append(cuboids_of_video[surr_idx][j + 1, k].normed_data(data_mean=mean, data_std=std))
                surroundings.append(cuboids_of_video[surr_idx][j + 1, k + 1].normed_data(data_mean=mean, data_std=std))
                surroundings.append(cuboids_of_video[surr_idx][j, k].normed_data(data_mean=mean, data_std=std))

                current_cuboid = cuboids_of_video[i][j, k].normed_data(data_mean=mean, data_std=std)

                loc_feats.append(sf.gen_local_feats_temporal(centre_cuboid=current_cuboid, surroundings=surroundings))

    return np.array(loc_feats)


def set_anomaly_status_local(list_cuboids, thresholds, mean_local, cov_inv_local, mean_data, std_data, past=False):
    alpha_local = thresholds[0]
    theta_local = thresholds[1]

    print "SETTING LOCAL STATUS"

    for i in tqdm(range(0, len(list_cuboids))):

        rows = list_cuboids[i][0].shape[0]
        cols = list_cuboids[i][0].shape[1]

        for l in range(1, len(list_cuboids[i])):
            for j in range(1, rows - 1):
                for k in range(1, cols - 1):
                    surroundings = []

                    if (past):
                        surr_idx = l - 1
                    else:
                        surr_idx = l

                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k - 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k + 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k - 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k + 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k - 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k + 1].normed_data(data_mean=mean_data, data_std=std_data))

                    current_cuboid = list_cuboids[i][l][j, k].normed_data(data_mean=mean_data, data_std=std_data)
                    past_cuboid = list_cuboids[i][l - 1][j, k].normed_data(data_mean=mean_data, data_std=std_data)

                    loc_feats = sf.gen_local_feats(centre_cuboid=current_cuboid, surroundings=surroundings,
                                                   past_cuboid=past_cuboid)
                    loc_feats = np.array(loc_feats)
                    loc_feats = loc_feats.reshape(1, loc_feats.shape[0])
                    loc_distance = models.make_distance(loc_feats, mean_local, cov_inv_local)

                    list_cuboids[i][l][j, k].update_anom_score_local(alpha_local, theta_local, loc_distance)

                    if (list_cuboids[i][l][j, k].anom_score >= 0.5):
                        # Set neighbours as possible anomalies
                        for m in range(max(0, l - 2), min(l + 3, len(list_cuboids[i]))):
                            for n in range(max(0, j - 2), min(j + 3, rows)):
                                for o in range(max(0, k - 2), min(k + 3, cols)):
                                    list_cuboids[i][m][n, o].update_local_possible_true()

    return list_cuboids


def set_anomaly_status_local_temporal(list_cuboids, thresholds, mean_local, cov_inv_local, mean_data, std_data,
                                      past=False):
    alpha_local = thresholds[0]
    theta_local = thresholds[1]

    print "SETTING LOCAL STATUS"

    for i in tqdm(range(0, len(list_cuboids))):

        rows = list_cuboids[i][0].shape[0]
        cols = list_cuboids[i][0].shape[1]

        for l in range(1, len(list_cuboids[i]) - 1):
            for j in range(1, rows - 1):
                for k in range(1, cols - 1):

                    surroundings = []

                    surr_idx = l - 1
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k - 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k + 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k - 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k + 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k - 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k + 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k].normed_data(data_mean=mean_data, data_std=std_data))

                    surr_idx = l + 1
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k - 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k + 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k - 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k + 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k - 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k + 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k].normed_data(data_mean=mean_data, data_std=std_data))

                    current_cuboid = list_cuboids[i][l][j, k].normed_data(data_mean=mean_data, data_std=std_data)

                    loc_feats = sf.gen_local_feats_temporal(centre_cuboid=current_cuboid, surroundings=surroundings)
                    loc_feats = np.array(loc_feats)
                    loc_feats = loc_feats.reshape(1, loc_feats.shape[0])
                    loc_distance = models.make_distance(loc_feats, mean_local, cov_inv_local)

                    list_cuboids[i][l][j, k].update_anom_score_local(alpha_local, theta_local, loc_distance)

                    if (list_cuboids[i][l][j, k].anom_score >= 0.5):
                        # Set neighbours as possible anomalies
                        for m in range(max(0, l - 1), min(l + 2, len(list_cuboids[i]))):
                            for n in range(max(0, j - 2), min(j + 3, rows)):
                                for o in range(max(0, k - 2), min(k + 3, cols)):
                                    list_cuboids[i][m][n, o].update_local_possible_true()

    return list_cuboids


def set_anomaly_status_global(list_cuboids, thresholds, model, mean_global,
                              cov_inv_global, mean_data, std_data):
    alpha_global = thresholds[0]
    theta_global = thresholds[1]

    print "SETTING GLOBAL STATUS"
    for i in tqdm(range(0, len(list_cuboids))):

        rows = list_cuboids[i][0].shape[0]
        cols = list_cuboids[i][0].shape[1]

        for l in range(0, len(list_cuboids[i])):
            for j in range(0, rows):
                for k in range(0, cols):
                    current_cuboid = list_cuboids[i][l][j, k].normed_data(data_mean=mean_data, data_std=std_data)

                    if (list_cuboids[i][l][j, k].anom_score >= 0.5 or list_cuboids[i][l][j, k].local_possible):

                        if (list_cuboids[i][l][j, k].local_possible and list_cuboids[i][l][j, k].anom_score == 0.0):
                            list_cuboids[i][l][j, k].anom_score = 0.5

                        global_feats = model.predict(np.expand_dims(current_cuboid, 0))
                        global_distance = models.make_distance(global_feats, mean_global, cov_inv_global)
                        list_cuboids[i][l][j, k].update_anom_score_global(alpha_global, theta_global, global_distance)

                    list_cuboids[i][l][j, k].update_status()

    return list_cuboids


def set_anomaly_status_global_single_th(list_cuboids, threshold, model, mean_global,
                                        cov_inv_global, mean_data, std_data):
    print "SETTING GLOBAL STATUS"
    for i in tqdm(range(0, len(list_cuboids))):

        rows = list_cuboids[i][0].shape[0]
        cols = list_cuboids[i][0].shape[1]

        for l in range(0, len(list_cuboids[i])):
            for j in range(0, rows):
                for k in range(0, cols):
                    current_cuboid = list_cuboids[i][l][j, k].normed_data(data_mean=mean_data, data_std=std_data)

                    if (list_cuboids[i][l][j, k].anom_score >= 0.5 or list_cuboids[i][l][j, k].local_possible):

                        if (list_cuboids[i][l][j, k].local_possible and list_cuboids[i][l][j, k].anom_score == 0.0):
                            list_cuboids[i][l][j, k].anom_score = 0.5

                        global_feats = model.predict(np.expand_dims(current_cuboid, 0))
                        global_distance = models.make_distance(global_feats, mean_global, cov_inv_global)

                        if (global_distance >= threshold):
                            list_cuboids[i][l][j, k].anom_score = 2.0

                    list_cuboids[i][l][j, k].update_status()

    return list_cuboids


def set_anomaly_status_global_mvg(list_cuboids, p_th, model, var, mean_data, std_data):
    print "SETTING GLOBAL STATUS"
    for i in tqdm(range(0, len(list_cuboids))):

        rows = list_cuboids[i][0].shape[0]
        cols = list_cuboids[i][0].shape[1]

        for l in range(0, len(list_cuboids[i])):
            for j in range(0, rows):
                for k in range(0, cols):
                    current_cuboid = list_cuboids[i][l][j, k].normed_data(data_mean=mean_data, data_std=std_data)

                    if (list_cuboids[i][l][j, k].anom_score >= 0.5 or list_cuboids[i][l][j, k].local_possible):

                        if (list_cuboids[i][l][j, k].local_possible and list_cuboids[i][l][j, k].anom_score == 0.0):
                            list_cuboids[i][l][j, k].anom_score = 0.5

                        global_feats = model.predict(np.expand_dims(current_cuboid, 0))
                        global_feats = global_feats[0]

                        lgpdf = var.logpdf(global_feats)

                        if (lgpdf <= p_th):
                            list_cuboids[i][l][j, k].anom_score = 2.0

                    list_cuboids[i][l][j, k].update_status()

    return list_cuboids


def set_anomaly_status_global_mahad_logpdf(list_cuboids, p_th, d_th, model1, model2, var, mean_data, std_data,
                                           mean_global, cov_inv_global):
    print "SETTING GLOBAL STATUS"
    for i in tqdm(range(0, len(list_cuboids))):

        rows = list_cuboids[i][0].shape[0]
        cols = list_cuboids[i][0].shape[1]

        for l in range(0, len(list_cuboids[i])):
            for j in range(0, rows):
                for k in range(0, cols):
                    current_cuboid = list_cuboids[i][l][j, k].normed_data(data_mean=mean_data, data_std=std_data)

                    # global_feats = model.predict(np.expand_dims(current_cuboid, 0))
                    global_feats = np.hstack((model1.predict(np.expand_dims(current_cuboid, 0)),
                                              model2.predict(np.expand_dims(current_cuboid, 0))))
                    global_distance = models.make_distance(global_feats, mean_global, cov_inv_global)
                    global_feats = global_feats[0]

                    lgpdf = var.logpdf(global_feats)

                    if (lgpdf <= p_th and global_distance >= d_th):
                        list_cuboids[i][l][j, k].anom_score = 2.0

                    list_cuboids[i][l][j, k].update_status()

    return list_cuboids


def plot_metrics(data_train, data_test, gt, title, x_label, y_label, filename, plt_name):
    list_colors = []
    for i in gt:

        if i == True:
            list_colors.append('red')

        elif i == False:
            list_colors.append('blue')
        else:
            print "##########################################"
            print "Some other value instead of TRUE or FALSE"
            print "##########################################"
    ax = plt.subplot(111)
    green_patch = mpatches.Patch(color='green', label='Metrics of Train Features')
    blue_patch = mpatches.Patch(color='blue', label='Metrics of Test Features (Not Anomalous)')
    red_patch = mpatches.Patch(color='red', label='Metrics of Test Features (Anomalous)')
    plt.scatter(range(0, len(data_train)), data_train, c='green', alpha=0.5)
    plt.scatter(range(0, len(data_test)), data_test, c=list_colors, alpha=0.5)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(handles=[green_patch, blue_patch, red_patch])
    plt.savefig(filename, bbox_inches='tight')
    with open(plt_name, 'wb') as fid:
        pickle.dump(ax, fid)

    plt.close()


def plot_hist(data, filename, title, x_label, y_label):
    # fixed bin size

    heights, bins = np.histogram(data, bins=100)
    plt.bar(bins[:-1], heights, width=(max(bins) - min(bins)) / len(bins), color="blue", alpha=0.5)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def plot_sepa(gt_array,filename,x_label,y_label,title,note):

    gt_array_values = np.zeros((len(gt_array)))
    gt_array_values[np.where(gt_array==True)] = 1.0
    list_colors=[]

    for i in gt_array:

        if i == True:
            list_colors.append('red')

        elif i == False:
            list_colors.append('blue')
        else:
            print "##########################################"
            print "Some other value instead of TRUE or FALSE"
            print "##########################################"

    f,ax = plt.subplots(figsize=(35,15))
    ax.set_ylim([-1,10])
    blue_patch = mpatches.Patch(color='blue', label='Not Anomalous')
    red_patch = mpatches.Patch(color='red', label='Anomalous')
    plt.scatter(range(0, len(gt_array)), gt_array_values, c=list_colors, alpha=0.5)
    plt.text(x=0,y=8,s=note)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(handles=[blue_patch, red_patch])
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def nn_metrics(path_videos, train_test, size_axis, n_frames, model, all_global_feats, mean_data, std_data, cov_mv_inv,
               verbose,normalize,fmin,fptp,pca,pca_obj):

    nn_obj = NearestNeighbors(n_neighbors=15, metric='mahalanobis', metric_params={'VI': cov_mv_inv}, n_jobs=-1)
    # kdensity = KernelDensity(metric='mahalanobis',metric_params={'VI':cov_mv_inv})
    # gaussianmixture = GaussianMixture(n_components=1000, verbose=1)

    print "########################"
    print "FITTING THE OBJECTS"
    print "########################"

    # print "Kdensity"
    # kdensity.fit(all_global_feats)

    print "Nearest Neighbours"
    nn_obj.fit(all_global_feats)

    with open(os.path.join('Results','nn_obj.pkl'), 'wb') as fid:
        pickle.dump(nn_obj, fid)

    # print "Gaussian Mixture"
    # gaussianmixture.fit(all_global_feats)

    nn_distance_train_list = []
    # kdensity_score_train_list = []
    # gaussianmixture_proba_train_list = []

    print "########################"
    print "STARTING THE LISTING"
    print "########################"

    for i in tqdm(range(0, len(all_global_feats))):
        dist, _ = nn_obj.kneighbors(np.expand_dims(all_global_feats[i], 0), return_distance=True)
        nn_distance_train_list.append(dist.mean())
        # kdensity_score_train_list.append(kdensity.score_samples(np.expand_dims(all_global_feats[i],0)))
        # gaussianmixture_proba_train_list.append(np.mean(gaussianmixture.predict_proba(np.expand_dims(all_global_feats[i],0))))

    nn_distance_train_list = np.array(nn_distance_train_list)
    # kdensity_score_train_list = np.array(kdensity_score_train_list)
    # gaussianmixture_proba_train_list = np.array(gaussianmixture_proba_train_list)

    nn_distance_train_list = np.sort(nn_distance_train_list)
    # kdensity_score_train_list = np.sort(kdensity_score_train_list)
    # gaussianmixture_proba_train_list = np.sort(gaussianmixture_proba_train_list)

    print "########################"
    print "START TEST CUBOIDS"
    print "########################"

    _, all_test_cuboids, all_test_gt = df.make_cuboids_of_videos(path_videos, train_test, size_axis, size_axis,
                                                                 n_frames)

    all_test_cuboids_normed = (all_test_cuboids - mean_data) / std_data

    del (all_test_cuboids)

    nn_distance_test_list = []
    # kdensity_score_test_list = []
    # gaussianmixture_proba_test_list=[]

    all_global_feats = model.predict(all_test_cuboids_normed, verbose=verbose)

    if(pca==True):
        print "############################################"
        print "PCA TRANSFORM OF TEST FEATS"
        print "############################################"
        all_global_feats = pca_obj.transform(all_global_feats)

    if(normalize==True):
        print "############################################"
        print "NORMALIZING TEST FEATS"
        print "############################################"

        all_global_feats = (all_global_feats - fmin)/fptp


    print ""
    print "########################"
    print "START TEST METRICS"
    print "########################"
    for i in tqdm(range(0, len(all_global_feats))):
        dist, _ = nn_obj.kneighbors(np.expand_dims(all_global_feats[i], 0), return_distance=True)
        nn_distance_test_list.append(dist.mean())
        # kdensity_score_test_list.append(kdensity.score_samples(np.expand_dims(all_global_feats[i],0)))
        # gaussianmixture_proba_test_list.append(np.mean(gaussianmixture.predict_proba(np.expand_dims(all_global_feats[i],0))))

    nn_distance_test_list = np.array(nn_distance_test_list)
    # kdensity_score_test_list = np.array(kdensity_score_test_list)
    # gaussianmixture_proba_test_list = np.array(gaussianmixture_proba_test_list)

    nn_distance_test_list_sort_args = np.argsort(nn_distance_test_list)
    nn_distance_test_list = nn_distance_test_list[nn_distance_test_list_sort_args]
    nn_distance_sorted_gt = all_test_gt[nn_distance_test_list_sort_args]
    del (nn_distance_test_list_sort_args)

    num_total_anomalies = np.sum(all_test_gt)
    least_score_of_anomaly_idx = len(all_test_gt) - num_total_anomalies
    least_score_of_anomaly = nn_distance_test_list[least_score_of_anomaly_idx]
    total_anomalies_detected = len(nn_distance_sorted_gt) - least_score_of_anomaly_idx
    total_anomalies_detected_percent= (len(nn_distance_sorted_gt) - least_score_of_anomaly_idx)/(len(nn_distance_sorted_gt)+0.0)
    true_positives = np.sum(nn_distance_sorted_gt[least_score_of_anomaly_idx:])
    false_positives = len(nn_distance_sorted_gt[least_score_of_anomaly_idx:])-true_positives

    precision = true_positives/(true_positives+false_positives+0.0)

    print "############################################"
    print " PRECISION NUMBERS:"
    print "############################################"
    print "Total number of anomalies:", num_total_anomalies
    print "Least score of anomaly - index", least_score_of_anomaly_idx
    print "Least score of anomaly:", least_score_of_anomaly
    print "total anomalies detected:",total_anomalies_detected
    print "fraction over thresholds:", total_anomalies_detected_percent
    print "true positives:", true_positives
    print "false_positives:", false_positives
    print "Precision:",precision
    print "###########################################"

    # kdensity_score_test_list_sort_args = np.argsort(kdensity_score_test_list)
    # kdensity_score_test_list = kdensity_score_test_list[kdensity_score_test_list_sort_args]
    # kdensity_score_sorted_gt = all_test_gt[kdensity_score_test_list_sort_args]
    # del(kdensity_score_test_list_sort_args)

    # gaussianmixture_proba_test_list_sort_args = np.argsort(gaussianmixture_proba_test_list)
    # gaussianmixture_proba_test_list = gaussianmixture_proba_test_list[gaussianmixture_proba_test_list_sort_args]
    # gaussianmixture_proba_sorted_gt = all_test_gt[gaussianmixture_proba_test_list_sort_args]
    # del(gaussianmixture_proba_test_list_sort_args)

    print ""
    print "############################"
    print "PLOTTING TRAIN-TEST METRICS"
    print "############################"

    plot_metrics(data_train=nn_distance_train_list, data_test=nn_distance_test_list, gt=nn_distance_sorted_gt,
                 title='10-NN DISTANCE OF TRAIN TEST FEATURES',
                 x_label='indices', y_label='10-nn distance sum values',
                 filename=os.path.join('Results', 'nnd_metrics.png'),
                 plt_name=os.path.join('Results', 'nnd_metrics.pkl'))

    # ff.plot_metrics(data_train=kdensity_score_train_list, data_test=kdensity_score_test_list,gt=kdensity_score_sorted_gt,title='KDENSITY SCORE OF TRAIN TEST',
    #                 x_label='indices',y_label='kdensity value',filename=os.path.join('Results','kdensity_metrics.png'),
    #                 plt_name=os.path.join('Results','kdensity_metrics.pkl'))
    #
    # ff.plot_metrics(data_train=gaussianmixture_proba_train_list, data_test=gaussianmixture_proba_test_list,gt=gaussianmixture_proba_sorted_gt,
    #                 title='GAUSSIAN MIXTURE PROB OF TRAIN TEST', x_label='indices',y_label='gaussian mixture prob value',filename=os.path.join('Results','gaussian_mixture_metrics.png'),
    #                 plt_name=os.path.join('Results','gaussian_mixture_metrics.pkl'))


    print "############################"
    print "PLOTTING TRAIN HISTOGRAMS"
    print "############################"

    plot_hist(nn_distance_train_list, os.path.join('Results', '10nn_distance_train_list.png'),
              x_label='10nn-sum of train cuboids',
              y_label='norm-count', title='Distribution of 10nn-sum of train features')

    # ff.plot_hist(kdensity_score_train_list,os.path.join('Results','kdensity_score_train_list.png'),x_label='kdensity score of train cuboids',
    #              y_label='norm-count',title='Kdensity score distribution - train cuboids')
    #
    # ff.plot_hist(gaussianmixture_proba_train_list,os.path.join('Results','gaussianmixture_proba_train_list.png'),x_label='gaussianmixture proba of train cuboids',
    #              y_label='norm-count',title='Gaussian Mixture Probability - train cuboids')


    print "############################"
    print "PLOTTING TEST HISTOGRAMS"
    print "############################"

    plot_hist(nn_distance_test_list, os.path.join('Results', '10nn_distance_test_list.png'),
              x_label='10nn-sum of test cuboids',
              y_label='norm-count', title='Distribution of 10nn-sum of test features')

    # ff.plot_hist(kdensity_score_test_list,os.path.join('Results','kdensity_score_test_list.png'),x_label='kdensity score of test cuboids',
    #              y_label='norm-count',title='Kdensity score distribution - test features')
    #
    # ff.plot_hist(gaussianmixture_proba_test_list,os.path.join('Results','gaussianmixture_proba_test_list.png'),x_label='gaussianmixture proba of test cuboids',
    #              y_label='norm-count',title='Gaussian Mixture Probability - test cuboids')

    print "############################"
    print "PLOTTING TEST SEPARATION METRICS"
    print "############################"
    plot_sepa(nn_distance_sorted_gt,os.path.join('Results','nn_dist_test_separation.png'),x_label='indices',
              y_label='anomaly-ground-truth', title='Ground truths over indices of the sorted nn-distance.'
              ,note= 'Note: Points to the right of any vertical line || to y-axis will be seen as anomalies')


    return True


def kdensity_metrics(path_videos, train_test, size_axis, n_frames, model, all_global_feats, mean_data, std_data,
                     cov_mv_inv, verbose,normalize,fmin,fptp,pca,pca_obj):

    kdensity = KernelDensity(metric='mahalanobis', metric_params={'VI': cov_mv_inv})

    print "########################"
    print "FITTING THE OBJECTS"
    print "########################"

    print "Kdensity"
    kdensity.fit(all_global_feats)

    with open(os.path.join('Results','kdensity_obj.pkl'), 'wb') as fid:
        pickle.dump(kdensity, fid)

    kdensity_score_train_list = []

    print "########################"
    print "STARTING THE LISTING"
    print "########################"

    for i in tqdm(range(0, len(all_global_feats))):
        kdensity_score_train_list.append(kdensity.score_samples(np.expand_dims(all_global_feats[i], 0)))

    kdensity_score_train_list = np.array(kdensity_score_train_list)
    kdensity_score_train_list = np.sort(kdensity_score_train_list)

    print "########################"
    print "START TEST CUBOIDS"
    print "########################"
    _, all_test_cuboids, all_test_gt = df.make_cuboids_of_videos(path_videos, train_test, size_axis, size_axis,
                                                                 n_frames)
    all_test_cuboids_normed = (all_test_cuboids - mean_data) / std_data

    del (all_test_cuboids)

    kdensity_score_test_list = []

    all_global_feats = model.predict(all_test_cuboids_normed, verbose=verbose)

    if(pca==True):
        print "############################################"
        print "PCA TRANSFORM OF TEST FEATS"
        print "############################################"
        all_global_feats = pca_obj.transform(all_global_feats)

    if(normalize==True):

        print "############################################"
        print "NORMALIZING TEST FEATS"
        print "############################################"

        all_global_feats = (all_global_feats - fmin)/fptp


    print ""
    print "########################"
    print "START TEST METRICS"
    print "########################"
    for i in tqdm(range(0, len(all_global_feats))):
        kdensity_score_test_list.append(kdensity.score_samples(np.expand_dims(all_global_feats[i], 0)))

    kdensity_score_test_list = np.array(kdensity_score_test_list)

    kdensity_score_test_list_sort_args = np.argsort(kdensity_score_test_list)
    kdensity_score_test_list = kdensity_score_test_list[kdensity_score_test_list_sort_args]
    kdensity_score_sorted_gt = all_test_gt[kdensity_score_test_list_sort_args]
    del (kdensity_score_test_list_sort_args)

    num_total_anomalies = np.sum(all_test_gt)
    least_score_of_anomaly_idx = len(all_test_gt) - num_total_anomalies
    least_score_of_anomaly = kdensity_score_test_list[least_score_of_anomaly_idx]
    total_anomalies_detected = len(kdensity_score_sorted_gt) - least_score_of_anomaly_idx
    total_anomalies_detected_percent= (len(kdensity_score_sorted_gt) - least_score_of_anomaly_idx)/(len(kdensity_score_sorted_gt)+0.0)
    true_positives = np.sum(kdensity_score_sorted_gt[least_score_of_anomaly_idx:])
    false_positives = len(kdensity_score_sorted_gt[least_score_of_anomaly_idx:])-true_positives

    precision = true_positives/(true_positives+false_positives+0.0)

    print "############################################"
    print " PRECISION NUMBERS:"
    print "############################################"
    print "Total number of anomalies:", num_total_anomalies
    print "Least score of anomaly - index", least_score_of_anomaly_idx
    print "Least score of anomaly:", least_score_of_anomaly
    print "total anomalies detected:",total_anomalies_detected
    print "fraction over thresholds:", total_anomalies_detected_percent
    print "true positives:", true_positives
    print "false_positives:", false_positives
    print "Precision:",precision
    print "###########################################"

    print ""
    print "############################"
    print "PLOTTING TRAIN-TEST METRICS"
    print "############################"

    plot_metrics(data_train=kdensity_score_train_list, data_test=kdensity_score_test_list, gt=kdensity_score_sorted_gt,
                 title='KDENSITY SCORE OF TRAIN TEST FEATURES',
                 x_label='indices', y_label='kdensity score', filename=os.path.join('Results', 'kdensity_metrics.png'),
                 plt_name=os.path.join('Results', 'kdensity_metrics.pkl'))

    print "############################"
    print "PLOTTING TRAIN HISTOGRAMS"
    print "############################"

    plot_hist(kdensity_score_train_list, os.path.join('Results', 'kdensity_score_train_list.png'),
              x_label='kdensity score of train cuboids',
              y_label='norm-count', title='Kernel density score distribution of features from train cuboids')

    print "############################"
    print "PLOTTING TEST HISTOGRAMS"
    print "############################"

    plot_hist(kdensity_score_test_list, os.path.join('Results', 'kdensity_score_test_list.png'),
              x_label='kdensity score of test cuboids',
              y_label='norm-count', title='Kernel density score distribution of features from test cuboids')

    print "############################"
    print "PLOTTING TEST SEPARATION METRICS"
    print "############################"
    plot_sepa(kdensity_score_sorted_gt,os.path.join('Results','kdensity_score_test_separation.png'),x_label='indices',
              y_label='anomaly-ground-truth', title='Ground truths over indices of the sorted kernel-density.'
              ,note= 'Note: Points to the right of any vertical line || to y-axis will be seen as anomalies')

    return True


def gaussianmixture_metrics(path_videos, train_test, size_axis, n_frames, model, all_global_feats, mean_data, std_data,
                            verbose,normalize,fmin,fptp,pca,pca_obj,n_comps):

    gaussianmixture = GaussianMixture(n_components=n_comps, verbose=1,max_iter=100000000)

    print "########################"
    print "FITTING THE OBJECTS"
    print "########################"

    print "Gaussian Mixture"
    gaussianmixture.fit(all_global_feats)

    with open(os.path.join('Results','gmixture_obj.pkl'), 'wb') as fid:
        pickle.dump(gaussianmixture, fid)

    gaussianmixture_proba_train_list = []

    print "########################"
    print "STARTING THE LISTING"
    print "########################"

    for i in tqdm(range(0, len(all_global_feats))):
        gaussianmixture_proba_train_list.append(-gaussianmixture.score(np.expand_dims(all_global_feats[i], 0)))

    gaussianmixture_proba_train_list = np.array(gaussianmixture_proba_train_list)
    gaussianmixture_proba_train_list = np.sort(gaussianmixture_proba_train_list)

    print "########################"
    print "START TEST CUBOIDS"
    print "########################"

    _, all_test_cuboids, all_test_gt = df.make_cuboids_of_videos(path_videos, train_test, size_axis, size_axis,
                                                                 n_frames)

    all_test_cuboids_normed = (all_test_cuboids - mean_data) / std_data

    del (all_test_cuboids)

    gaussianmixture_proba_test_list = []

    all_global_feats = model.predict(all_test_cuboids_normed, verbose=verbose)

    if(pca==True):
        print "############################################"
        print "PCA TRANSFORM OF TEST FEATS"
        print "############################################"
        all_global_feats = pca_obj.transform(all_global_feats)


    if(normalize==True):

        print "############################################"
        print "NORMALIZING TEST FEATS"
        print "############################################"

        all_global_feats = (all_global_feats - fmin)/fptp


    print ""
    print "########################"
    print "START TEST METRICS"
    print "########################"
    for i in tqdm(range(0, len(all_global_feats))):
        gaussianmixture_proba_test_list.append(-gaussianmixture.score(np.expand_dims(all_global_feats[i], 0)))

    gaussianmixture_proba_test_list = np.array(gaussianmixture_proba_test_list)

    gaussianmixture_proba_test_list_sort_args = np.argsort(gaussianmixture_proba_test_list)
    gaussianmixture_proba_test_list = gaussianmixture_proba_test_list[gaussianmixture_proba_test_list_sort_args]
    gaussianmixture_proba_sorted_gt = all_test_gt[gaussianmixture_proba_test_list_sort_args]
    del (gaussianmixture_proba_test_list_sort_args)

    num_total_anomalies = np.sum(all_test_gt)
    least_score_of_anomaly_idx = len(all_test_gt) - num_total_anomalies
    least_score_of_anomaly = gaussianmixture_proba_test_list[least_score_of_anomaly_idx]
    total_anomalies_detected = len(gaussianmixture_proba_sorted_gt) - least_score_of_anomaly_idx
    total_anomalies_detected_percent= (len(gaussianmixture_proba_sorted_gt) - least_score_of_anomaly_idx)/(len(gaussianmixture_proba_sorted_gt)+0.0)
    true_positives = np.sum(gaussianmixture_proba_sorted_gt[least_score_of_anomaly_idx:])
    false_positives = len(gaussianmixture_proba_sorted_gt[least_score_of_anomaly_idx:])-true_positives

    precision = true_positives/(true_positives+false_positives+0.0)

    print "############################################"
    print " PRECISION NUMBERS:"
    print "############################################"
    print "Total number of anomalies:", num_total_anomalies
    print "Least score of anomaly - index", least_score_of_anomaly_idx
    print "Least score of anomaly:", least_score_of_anomaly
    print "total anomalies detected:",total_anomalies_detected
    print "fraction over thresholds:", total_anomalies_detected_percent
    print "true positives:", true_positives
    print "false_positives:", false_positives
    print "Precision:",precision
    print "###########################################"

    print ""
    print "############################"
    print "PLOTTING TRAIN-TEST METRICS"
    print "############################"

    plot_metrics(data_train=gaussianmixture_proba_train_list, data_test=gaussianmixture_proba_test_list,
                 gt=gaussianmixture_proba_sorted_gt,
                 title='GAUSSIAN MIXTURE ('+ str(n_comps)+' components) LIKELIHOOD SCORE OF TRAIN TEST FEATURES', x_label='indices',
                 y_label='Gaussian mixture likelihood value',
                 filename=os.path.join('Results', 'gaussian_mixture_metrics'+str(n_comps)+'.png'),
                 plt_name=os.path.join('Results', 'gaussian_mixture_metrics.pkl'))

    print "############################"
    print "PLOTTING TRAIN HISTOGRAMS"
    print "############################"

    plot_hist(gaussianmixture_proba_train_list, os.path.join('Results', 'gaussianmixture_proba_train_list'+str(n_comps)+'.png'),
              x_label='gaussianmixture proba of train cuboids',
              y_label='norm-count', title='Likelihood score of features from train cuboids')

    print "############################"
    print "PLOTTING TEST HISTOGRAMS"
    print "############################"

    plot_hist(gaussianmixture_proba_test_list, os.path.join('Results', 'gaussianmixture_proba_test_list'+str(n_comps)+'.png'),
              x_label='gaussianmixture proba of test cuboids',
              y_label='norm-count', title='Likelihood score of features from test cuboids')

    print "############################"
    print "PLOTTING TEST SEPARATION METRICS"
    print "############################"
    plot_sepa(gaussianmixture_proba_sorted_gt,os.path.join('Results','gaussianmixture_test_separation'+str(n_comps)+'.png'),x_label='indices',
              y_label='anomaly-ground-truth', title='Ground truths over indices of the sorted gaussianmixture_scores.'
              ,note= 'Note: Points to the right of any vertical line || to y-axis will be seen as anomalies')

    return True

def kmeans_clustering(path_videos, train_test, size_axis, n_frames, model, all_global_feats, mean_data, std_data,
                            verbose,normalize,fmin,fptp,pca,pca_obj,n_comps):

    km = KMeans(n_clusters=n_comps, verbose=1,n_jobs=-1)

    print "########################"
    print "FITTING THE OBJECTS"
    print "########################"

    print "K Means"
    km.fit(all_global_feats)

    with open(os.path.join('Results','kmeans_obj.pkl'), 'wb') as fid:
        joblib.dump(km, fid)

    kmeans_proba_train_list = []

    print "########################"
    print "STARTING THE LISTING"
    print "########################"

    for i in tqdm(range(0, len(all_global_feats))):
        kmeans_proba_train_list.append(km.transform(np.expand_dims(all_global_feats[i], 0)).min(axis=1)[0])

    kmeans_proba_train_list = np.array(kmeans_proba_train_list)
    kmeans_proba_train_list = np.sort(kmeans_proba_train_list)

    print "########################"
    print "START TEST CUBOIDS"
    print "########################"

    _, all_test_cuboids, all_test_gt = df.make_cuboids_of_videos(path_videos, train_test, size_axis, size_axis,
                                                                 n_frames)

    all_test_cuboids_normed = (all_test_cuboids - mean_data) / std_data

    del (all_test_cuboids)

    kmeans_proba_test_list = []

    all_global_feats = model.predict(all_test_cuboids_normed, verbose=verbose)

    if(pca==True):
        print "############################################"
        print "PCA TRANSFORM OF TEST FEATS"
        print "############################################"
        all_global_feats = pca_obj.transform(all_global_feats)


    if(normalize==True):

        print "############################################"
        print "NORMALIZING TEST FEATS"
        print "############################################"

        all_global_feats = (all_global_feats - fmin)/fptp


    print ""
    print "########################"
    print "START TEST METRICS"
    print "########################"
    for i in tqdm(range(0, len(all_global_feats))):
        kmeans_proba_test_list.append(km.transform(np.expand_dims(all_global_feats[i], 0)).min(axis=1)[0])

    kmeans_proba_test_list = np.array(kmeans_proba_test_list)

    kmeans_proba_test_list_sort_args = np.argsort(kmeans_proba_test_list)
    kmeans_proba_test_list = kmeans_proba_test_list[kmeans_proba_test_list_sort_args]
    kmeans_proba_sorted_gt = all_test_gt[kmeans_proba_test_list_sort_args]
    del (kmeans_proba_test_list_sort_args)

    num_total_anomalies = np.sum(all_test_gt)
    least_score_of_anomaly_idx = len(all_test_gt) - num_total_anomalies
    least_score_of_anomaly = kmeans_proba_test_list[least_score_of_anomaly_idx]
    total_anomalies_detected = len(kmeans_proba_sorted_gt) - least_score_of_anomaly_idx
    total_anomalies_detected_percent= (len(kmeans_proba_sorted_gt) - least_score_of_anomaly_idx)/(len(kmeans_proba_sorted_gt)+0.0)
    true_positives = np.sum(kmeans_proba_sorted_gt[least_score_of_anomaly_idx:])
    false_positives = len(kmeans_proba_sorted_gt[least_score_of_anomaly_idx:])-true_positives

    precision = true_positives/(true_positives+false_positives+0.0)

    print "############################################"
    print " PRECISION NUMBERS:"
    print "############################################"
    print "Total number of anomalies:", num_total_anomalies
    print "Least score of anomaly - index", least_score_of_anomaly_idx
    print "Least score of anomaly:", least_score_of_anomaly
    print "total anomalies detected:",total_anomalies_detected
    print "fraction over thresholds:", total_anomalies_detected_percent
    print "true positives:", true_positives
    print "false_positives:", false_positives
    print "Precision:",precision
    print "###########################################"

    print ""
    print "############################"
    print "PLOTTING TRAIN-TEST METRICS"
    print "############################"

    plot_metrics(data_train=kmeans_proba_train_list, data_test=kmeans_proba_test_list,
                 gt=kmeans_proba_sorted_gt,
                 title='kmeans ('+ str(n_comps)+' components) KMEANS SCORE OF TRAIN TEST FEATURES', x_label='indices',
                 y_label='Kmeans likelihood value',
                 filename=os.path.join('Results', 'kmeans_metrics'+str(n_comps)+'.png'),
                 plt_name=os.path.join('Results', 'kmeans_metrics.pkl'))

    print "############################"
    print "PLOTTING TRAIN HISTOGRAMS"
    print "############################"

    plot_hist(kmeans_proba_train_list, os.path.join('Results', 'kmeans_proba_train_list'+str(n_comps)+'.png'),
              x_label='kmeans proba of train cuboids',
              y_label='norm-count', title='Likelihood score of features from train cuboids')

    print "############################"
    print "PLOTTING TEST HISTOGRAMS"
    print "############################"

    plot_hist(kmeans_proba_test_list, os.path.join('Results', 'kmeans_proba_test_list'+str(n_comps)+'.png'),
              x_label='kmeans proba of test cuboids',
              y_label='norm-count', title='Likelihood score of features from test cuboids')

    print "############################"
    print "PLOTTING TEST SEPARATION METRICS"
    print "############################"
    plot_sepa(kmeans_proba_sorted_gt,os.path.join('Results','kmeans_test_separation'+str(n_comps)+'.png'),x_label='indices',
              y_label='anomaly-ground-truth', title='Ground truths over indices of the sorted kmeans_scores.'
              ,note= 'Note: Points to the right of any vertical line || to y-axis will be seen as anomalies')

    return True

def logpdf_mertrics(path_videos, train_test, size_axis, n_frames, model, all_global_feats, mean_data, std_data,
                    verbose,normalize,fmin,fptp,pca,pca_obj):

    mean_mvg = np.mean(all_global_feats, axis=0)
    cov_mvg = np.cov(all_global_feats, rowvar=False)

    var = multivariate_normal(mean=mean_mvg, cov=cov_mvg)

    print "MAKING LOG-PDF DISTRIBUTION OF TRAIN CUBOIDS"
    log_pdf_train_list = []


    for i in tqdm(range(0, len(all_global_feats))):
        log_pdf_train_list.append(-var.logpdf(all_global_feats[i]))

    log_pdf_train_list = np.array(log_pdf_train_list)

    log_pdf_train_list = np.sort(log_pdf_train_list)

    print "START TEST CUBOIDS"

    _, all_test_cuboids, all_test_gt = df.make_cuboids_of_videos(path_videos, train_test, size_axis, size_axis,
                                                                 n_frames)
    all_test_cuboids_normed = (all_test_cuboids - mean_data) / std_data
    del (all_test_cuboids)

    log_pdf_test_list = []

    # all_global_feats = np.hstack((model1.predict(all_test_cuboids_normed, verbose=1),model2.predict(all_test_cuboids_normed,verbose=1)))

    all_global_feats = model.predict(all_test_cuboids_normed, verbose=verbose)

    if(pca==True):
        print "############################################"
        print "PCA TRANSFORM OF TEST FEATS"
        print "############################################"
        all_global_feats = pca_obj.transform(all_global_feats)


    if(normalize==True):

        print "############################################"
        print "NORMALIZING TEST FEATS"
        print "############################################"

        all_global_feats = (all_global_feats - fmin)/fptp


    for i in tqdm(range(0, len(all_global_feats))):
        log_pdf_test_list.append(-var.logpdf(all_global_feats[i]))

    log_pdf_test_list = np.array(log_pdf_test_list)

    all_test_gt = np.array(all_test_gt)

    log_pdf_test_list_sort_args = np.argsort(log_pdf_test_list)
    log_pdf_test_list = log_pdf_test_list[log_pdf_test_list_sort_args]
    log_pdf_sorted_gt = all_test_gt[log_pdf_test_list_sort_args]
    del (log_pdf_test_list_sort_args)

    num_total_anomalies = np.sum(all_test_gt)
    least_score_of_anomaly_idx = len(all_test_gt) - num_total_anomalies
    least_score_of_anomaly = log_pdf_test_list[least_score_of_anomaly_idx]
    total_anomalies_detected = len(log_pdf_sorted_gt) - least_score_of_anomaly_idx
    total_anomalies_detected_percent = (len(log_pdf_sorted_gt) - least_score_of_anomaly_idx) / (len(log_pdf_sorted_gt)+ 0.0)
    true_positives = np.sum(log_pdf_sorted_gt[least_score_of_anomaly_idx:])
    false_positives = len(log_pdf_sorted_gt[least_score_of_anomaly_idx:]) - true_positives

    precision = true_positives / (true_positives + false_positives + 0.0)

    print "############################################"
    print " PRECISION NUMBERS:"
    print "############################################"
    print "Total number of anomalies:", num_total_anomalies
    print "Least score of anomaly - index", least_score_of_anomaly_idx
    print "Least score of anomaly:", least_score_of_anomaly
    print "total anomalies detected:", total_anomalies_detected
    print "fraction over thresholds:", total_anomalies_detected_percent
    print "true positives:", true_positives
    print "false_positives:", false_positives
    print "Precision:", precision
    print "###########################################"

    del (all_test_gt)

    print ""
    print "############################"
    print "PLOTTING TRAIN-TEST METRICS"
    print "############################"

    plot_metrics(data_train=log_pdf_train_list, data_test=log_pdf_test_list, gt=log_pdf_sorted_gt,
                 title='NEG. LOG GAUSSIAN PDF OF TRAIN & TEST FEATURES ',
                 x_label='indices', y_label='negative log gaussian pdf value',
                 filename=os.path.join('Results', 'logpdf_metrics.png'),
                 plt_name=os.path.join('Results', 'logpdf_metrics.pkl'))

    print "############################"
    print "PLOTTING TRAIN HISTOGRAMS"
    print "############################"

    plot_hist(log_pdf_train_list, os.path.join('Results', 'log_pdf_train_list.png'),
              x_label='neg.log gaussian pdf of train cuboids',
              y_label='norm-count',
              title='Distribution of Negative Log Gaussian Probability of features from train cuboids')

    print "############################"
    print "PLOTTING TEST HISTOGRAMS"
    print "############################"

    plot_hist(log_pdf_test_list, os.path.join('Results', 'log_pdf_test_list.png'), x_label='neg.log gaussian pdf of test cuboids',
              y_label='norm-count',
              title='Distribution of Negative Log Gaussian Probability of features from test cuboids')

    plot_sepa(log_pdf_sorted_gt,os.path.join('Results','log_pdf_test_separation.png'),x_label='indices',
              y_label='anomaly-ground-truth', title='Ground truths over indices of the sorted neg-log-pdfs.'
              ,note= 'Note: Points to the right of any vertical line || to y-axis will be seen as anomalies')


    return True


def mahad_metrics(path_videos, train_test, size_axis, n_frames, model, all_global_feats, mean_data, std_data, verbose,normalize,fmin,fptp,pca,pca_obj):

    mean_mvg = np.mean(all_global_feats, axis=0)
    cov_mvg = np.cov(all_global_feats, rowvar=False)
    cov_mv_inv = np.linalg.inv(cov_mvg)

    print "MAKING MAHAD DISTRIBUTION OF TRAIN CUBOIDS"
    maha_dist_train_list = []

    mean = np.reshape(mean_mvg, (1, mean_mvg.shape[0]))

    for i in tqdm(range(0, len(all_global_feats))):
        feats = np.reshape(all_global_feats[i], (1, all_global_feats[i].shape[0]))
        maha_dist_train_list.append(mdls.make_distance(feats, mean, cov_mv_inv))

    maha_dist_train_list = np.array(maha_dist_train_list)

    maha_dist_train_list = np.sort(maha_dist_train_list)

    print "START TEST CUBOIDS"

    _, all_test_cuboids, all_test_gt = df.make_cuboids_of_videos(path_videos, train_test, size_axis, size_axis,
                                                                 n_frames)

    all_test_cuboids_normed = (all_test_cuboids - mean_data) / std_data

    del (all_test_cuboids)

    maha_dist_test_list = []

    # all_global_feats = np.hstack((model1.predict(all_test_cuboids_normed, verbose=1),model2.predict(all_test_cuboids_normed,verbose=1)))

    all_global_feats = model.predict(all_test_cuboids_normed, verbose=verbose)

    if(pca==True):
        print "############################################"
        print "PCA TRANSFORM OF TEST FEATS"
        print "############################################"
        all_global_feats = pca_obj.transform(all_global_feats)

    if(normalize==True):

        print "############################################"
        print "NORMALIZING TEST FEATS"
        print "############################################"

        all_global_feats = (all_global_feats - fmin)/fptp

    mean = np.reshape(mean_mvg, (1, mean_mvg.shape[0]))

    for i in tqdm(range(0, len(all_global_feats))):
        feats = np.reshape(all_global_feats[i], (1, all_global_feats[i].shape[0]))
        maha_dist_test_list.append(mdls.make_distance(feats, mean, cov_mv_inv))

    maha_dist_test_list = np.array(maha_dist_test_list)

    all_test_gt = np.array(all_test_gt)

    maha_dist_test_list_sort_args = np.argsort(maha_dist_test_list)
    maha_dist_test_list = maha_dist_test_list[maha_dist_test_list_sort_args]
    maha_dist_sorted_gt = all_test_gt[maha_dist_test_list_sort_args]

    num_total_anomalies = np.sum(all_test_gt)
    least_score_of_anomaly_idx = len(all_test_gt) - num_total_anomalies
    least_score_of_anomaly = maha_dist_test_list[least_score_of_anomaly_idx]
    total_anomalies_detected = len(maha_dist_sorted_gt) - least_score_of_anomaly_idx
    total_anomalies_detected_percent= (len(maha_dist_sorted_gt) - least_score_of_anomaly_idx)/(len(maha_dist_sorted_gt)+0.0)
    true_positives = np.sum(maha_dist_sorted_gt[least_score_of_anomaly_idx:])
    false_positives = len(maha_dist_sorted_gt[least_score_of_anomaly_idx:])-true_positives

    precision = true_positives/(true_positives+false_positives+0.0)

    print "############################################"
    print " PRECISION NUMBERS:"
    print "############################################"
    print "Total number of anomalies:", num_total_anomalies
    print "Least score of anomaly - index", least_score_of_anomaly_idx
    print "Least score of anomaly:", least_score_of_anomaly
    print "total anomalies detected:",total_anomalies_detected
    print "fraction over thresholds:", total_anomalies_detected_percent
    print "true positives:", true_positives
    print "false_positives:", false_positives
    print "Precision:",precision
    print "###########################################"

    del (maha_dist_test_list_sort_args)

    del (all_test_gt)

    print ""
    print "############################"
    print "PLOTTING TRAIN-TEST METRICS"
    print "############################"

    plot_metrics(data_train=maha_dist_train_list, data_test=maha_dist_test_list, gt=maha_dist_sorted_gt,
                 title='MAHALANOBIS DISTANCE OF TRAIN TEST FEATURES FROM MEAN',
                 x_label='indices', y_label='mahalanobis distance value from mean',
                 filename=os.path.join('Results', 'mahad_metrics.png'),
                 plt_name=os.path.join('Results', 'mahad_metrics.pkl'))

    print "############################"
    print "PLOTTING TRAIN HISTOGRAMS"
    print "############################"

    plot_hist(maha_dist_train_list, os.path.join('Results', 'maha_dist_train_list.png'),
              x_label='maha dist of train cuboids',
              y_label='norm-count', title='Mahalanobis distance distribution of features from train cuboids ')

    print "############################"
    print "PLOTTING TEST HISTOGRAMS"
    print "############################"
    plot_hist(maha_dist_test_list, os.path.join('Results', 'maha_dist_test_list.png'),
              x_label='maha dist of test cuboids',
              y_label='norm-count', title='Mahalanobis distance distribution of features from test cuboids')

    print "############################"
    print "PLOTTING TEST SEPARATION METRICS"
    print "############################"
    plot_sepa(maha_dist_sorted_gt,os.path.join('Results','maha_dist_test_separation.png'),x_label='indices',
              y_label='anomaly-ground-truth', title='Ground truths over indices of the sorted maha-d.'
              ,note= 'Note: Points to the right of any vertical line || to y-axis will be seen as anomalies')

    return True

def lexsort_based_unique(data):
    sorted_data =  data[np.lexsort(data.T),:]
    row_mask = np.append([True],np.any(np.diff(sorted_data,axis=0),1))
    return sorted_data[row_mask]

def make_dictionary(list_cuboids, kmeans_obj, model,mean_data, std_data,mean_feats,std_feats):

    print "$$$$$$$$$$$$$$$$$$$$$$$$"
    print "GENERATING DICTIONARY"
    print "$$$$$$$$$$$$$$$$$$$$$$$$"

    word_list = []
    for i in tqdm(range(0, len(list_cuboids))):

        rows = list_cuboids[i][0].shape[0]
        cols = list_cuboids[i][0].shape[1]

        for l in range(1, len(list_cuboids[i]) - 1):
            for j in range(1, rows - 1):
                for k in range(1, cols - 1):

                    surroundings = []

                    surr_idx = l - 1
                    current_cuboid = list_cuboids[i][l][j, k].normed_data(data_mean=mean_data, data_std=std_data)

                    surroundings.append(current_cuboid)

                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k - 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k + 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k - 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k + 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k - 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k + 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k].normed_data(data_mean=mean_data, data_std=std_data))

                    surr_idx = l
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k - 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k + 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k - 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k + 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k - 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k + 1].normed_data(data_mean=mean_data, data_std=std_data))


                    surr_idx = l + 1
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k - 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k + 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k - 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k + 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k - 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k + 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k].normed_data(data_mean=mean_data, data_std=std_data))

                    data_array = np.array(surroundings)
                    surrounding_feats = model.predict(data_array)
                    surrounding_feats = (surrounding_feats - mean_feats)/std_feats
                    word = kmeans_obj.predict(surrounding_feats)

                    word_list.append(word.tolist())



    return word_list

def dictionary_based_anom_setting(list_cuboids,model,kmeans_obj,dictionary,mean_data, std_data,mean_feats,std_feats):

    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
    print "GENERATING DICTIONARY BASED ANOMALY SETTINGS"
    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

    list_of_frequencies = [0]*len(dictionary)

    for i in tqdm(range(0, len(list_cuboids))):

        rows = list_cuboids[i][0].shape[0]
        cols = list_cuboids[i][0].shape[1]

        for l in range(1, len(list_cuboids[i]) - 1):
            for j in range(1, rows - 1):
                for k in range(1, cols - 1):

                    surroundings = []

                    surr_idx = l - 1
                    current_cuboid = list_cuboids[i][l][j, k].normed_data(data_mean=mean_data, data_std=std_data)

                    surroundings.append(current_cuboid)

                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k - 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k + 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k - 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k + 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k - 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k + 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k].normed_data(data_mean=mean_data, data_std=std_data))

                    surr_idx = l
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k - 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k + 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k - 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k + 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k - 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k + 1].normed_data(data_mean=mean_data, data_std=std_data))


                    surr_idx = l + 1
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k - 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k + 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k - 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k + 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k - 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k + 1].normed_data(data_mean=mean_data, data_std=std_data))
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k].normed_data(data_mean=mean_data, data_std=std_data))

                    data_array = np.array(surroundings)
                    surrounding_feats = model.predict(data_array)
                    surrounding_feats = (surrounding_feats - mean_feats) / std_feats
                    word = kmeans_obj.predict(surrounding_feats)
                    word = word.tolist()

                    if(word in dictionary):
                        idx = dictionary.index(word)
                        # print "word:",word, "matches with ", dictionary[idx]
                        list_of_frequencies[idx]=list_of_frequencies[idx]+1
                    else:
                        # print "word:",word, " does not match with any dictionary entry."
                        list_cuboids[i][l][j, k].anom_score = 2.0

                    list_cuboids[i][l][j, k].update_status()


    return list_cuboids, list_of_frequencies