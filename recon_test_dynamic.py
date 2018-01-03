import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import Model
import os
import numpy as np
from data_pkg import data_fns as df
from model_pkg import models
import pickle

print "LOAD TEST LIST_CUBOIDS and mean,std"

path_videos = '/usr/local/data/sejacob/ANOMALY/data/UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test'
train_test = 'Test'
size_axis = 24
n_frames = 11
list_cuboids_test, _ = df.make_cuboids_of_videos(path_videos, train_test, size_axis, size_axis, n_frames,num=1)


mean = np.load(os.path.join('data_stored_temporal', 'cuboid_train_mean.npy'))
std = np.load(os.path.join('data_stored_temporal', 'cuboid_train_std.npy'))

one_array_cuboids = list_cuboids_test[0][120]

print "deleting the list to save ram-space"
del (list_cuboids_test)

print one_array_cuboids.shape

size_axis = 24
n_frames = 11
print "GETTING THE MODEL"
model = models.small_2d_conv_net_dynamic(size_y=size_axis, size_x=size_axis, n_channels=1, n_frames=n_frames,filters=20)
model.summary()
model_name = 'cnn_model_dynamic.h5'

model.load_weights(os.path.join('saved_models', model_name))
model = Model(inputs=model.input, outputs=model.layers[2].output)

n_frames_dynamics = model.output_shape[-1]

images_recon = np.zeros((size_axis * one_array_cuboids.shape[0], size_axis * one_array_cuboids.shape[1], n_frames_dynamics))
images = np.zeros((size_axis * one_array_cuboids.shape[0], size_axis * one_array_cuboids.shape[1], 1, n_frames))

for i in range(0, one_array_cuboids.shape[0]):
    for j in range(0, one_array_cuboids.shape[1]):
        images[size_axis * i:size_axis * (i + 1), size_axis * j:size_axis * (j + 1), :, :] = one_array_cuboids[i][j].data

for i in range(0, one_array_cuboids.shape[0]):
    for j in range(0, one_array_cuboids.shape[1]):

        array_to_be_predicted = np.expand_dims(one_array_cuboids[i][j].normed_data(mean, std),axis=0)
        predicted = model.predict(array_to_be_predicted)[0]
        predicted_scaled = ((predicted - predicted.min())/(predicted.max()-predicted.min())) * 255.0
        images_recon[size_axis * i:size_axis * (i + 1), size_axis * j:size_axis * (j + 1), :] = np.uint8(predicted_scaled)



print "SAVING RECON IMAGES"

for i in range(0, n_frames):

    plt.imshow(images[:, :, 0, i],cmap='Greys_r')
    plt.savefig(os.path.join('Results','recon_images','real_image_'+str(i)+'.png'))
    plt.close()


for i in range(0, n_frames_dynamics):

    plt.imshow(images_recon[:, :, i],cmap='Greys_r')
    plt.savefig(os.path.join('Results','recon_images','dyn_image_'+str(i)+'.png'))
    plt.close()

