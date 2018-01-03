import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from keras.models import load_model
from model_pkg import models
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os

path_videos = '/usr/local/data/sejacob/ANOMALY/data/UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train'
train_test = 'Train'

print "LOADING CUBOIDS AND FEATS"

folder = 'data_stored_temporal'
all_cuboids_normed = np.load(os.path.join(folder, 'all_cuboids_normed.npy'))

size_axis = 12
n_frames = 5

print "GETTING THE MODEL"
model_name = 'cnn_model_32_hid.h5'
model = models.small_2d_conv_net(size_y=size_axis, size_x=size_axis, n_channels=1, n_frames=n_frames,h_units=32)
model.load_weights(os.path.join('saved_models', model_name))
model.summary()


print "TESTING THE TRAINING LOSS"

mcp = ModelCheckpoint(os.path.join('saved_models', model_name), save_best_only='True',verbose=1,save_weights_only=True)
es = EarlyStopping(min_delta=1e-5, patience=10, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

history = model.fit(all_cuboids_normed, all_cuboids_normed, verbose=1, validation_split=0.2, batch_size=1024,
                    shuffle=True, epochs=200, callbacks=[mcp, es, rlr])

del(model)

model_name = 'cnn_model_32_hid.h5'
model = models.small_2d_conv_net(size_y=size_axis, size_x=size_axis, n_channels=1, n_frames=n_frames,h_units=32)
model.load_weights(os.path.join('saved_models', model_name))
model.fit(all_cuboids_normed,all_cuboids_normed,batch_size=512,epochs=5)
model.save_weights(os.path.join('saved_models', model_name))

#summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.savefig(os.path.join('graphs', 'plot_train_' + model_name+ '.png'))
plt.close()
