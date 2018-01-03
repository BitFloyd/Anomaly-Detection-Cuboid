from keras.models import Model
from keras.layers import Dense, Input, Flatten, Reshape, Dropout
from keras.layers import Conv2D,MaxPooling2D,UpSampling2D,LeakyReLU,BatchNormalization,GaussianNoise,SpatialDropout2D
from keras.optimizers import RMSprop
from keras.regularizers import l1,l2,l1_l2
from scipy.spatial.distance import cdist
import numpy as np
from keras import backend as K
from keras.engine.topology import Layer
from theano.tensor.nnet.neighbours import images2neibs



class DynImage(Layer):

    def __init__(self, filters, **kwargs):
        self.filters = filters
        super(DynImage, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.filters,input_shape[-1]),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(DynImage, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.dot(x, K.transpose(self.kernel))

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0],input_shape[1],input_shape[2],self.filters)
        return output_shape

def loss_DSSIM_theano(y_true, y_pred):
    # There are additional parameters for this function
    # Note: some of the 'modes' for edge behavior do not yet have a gradient definition in the Theano tree
    # and cannot be used for learning
    y_true = y_true.dimshuffle([0, 3, 1, 2])
    y_pred = y_pred.dimshuffle([0, 3, 1, 2])
    patches_true = images2neibs(y_true, [4, 4])
    patches_pred = images2neibs(y_pred, [4, 4])

    u_true = K.mean(patches_true, axis=-1)
    u_pred = K.mean(patches_pred, axis=-1)

    var_true = K.var(patches_true, axis=-1)
    var_pred = K.var(patches_pred, axis=-1)
    std_true = K.sqrt(var_true + K.epsilon())
    std_pred = K.sqrt(var_pred + K.epsilon())

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    ssim /= denom  # no need for clipping, c1 and c2 make the denom non-zero

    return K.sum(1.0 - ssim)


def one_layer_sparse_ac(size_y=11, size_x=11, n_channels=1, n_frames=5, h_units=1000, l1_term=0.01, weight_l2_term=0.01):
    inp = Input(shape=(size_y, size_x, n_channels, n_frames))
    x1 = Flatten()(inp)
    # x1 = Dropout(0.2)(x1)
    x2 = Dense(units=h_units, activation='sigmoid', activity_regularizer=l1(l1_term))(x1)
    y1 = Dense(units=(size_y * size_x * n_channels * n_frames), activation='linear', kernel_regularizer=l2(weight_l2_term))(x2)
    y2 = Reshape(target_shape=(size_y, size_x, n_channels, n_frames))(y1)

    model = Model(inputs=inp, outputs=y2)

    rmsprop = RMSprop(lr=0.0001)

    model.compile(optimizer=rmsprop, loss='mse')

    return model

def four_layer_ac(size_y=11, size_x=11, n_channels=1, n_frames=5, h_units=100, weight_l1=0.01,weight_l2=0.01):

    inp = Input(shape=(size_y, size_x, n_channels, n_frames))

    x1 = Flatten()(inp)
    x1 = Dropout(0.3)(x1)
    x2 = Dense(units=h_units*2, activation='tanh', kernel_regularizer=l1_l2(l2=weight_l2))(x1)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.3)(x2)
    x3 = Dense(units=h_units,activation='tanh', kernel_regularizer=l1_l2(l2=weight_l2))(x2)
    x3 = BatchNormalization()(x3)
    x3 = Dropout(0.3)(x3)

    y1 = Dense(units=h_units*2,activation='tanh', kernel_regularizer=l1_l2(l2=weight_l2))(x3)
    y1 = BatchNormalization()(y1)
    y1 = Dropout(0.3)(y1)
    y1 = Dense(units=(size_y * size_x * n_channels * n_frames), activation='tanh', kernel_regularizer=l1_l2(l1=weight_l1, l2=weight_l2))(y1)
    y2 = Reshape(target_shape=(size_y, size_x, n_channels, n_frames))(y1)

    model = Model(inputs=inp, outputs=y2)

    rmsprop = RMSprop(lr=0.0001)

    model.compile(optimizer=rmsprop, loss='mse')

    return model

def small_2d_conv_net(size_y=32,size_x=32,n_channels=1,n_frames=5,h_units=100):

    inp = Input(shape=(size_y,size_x,n_channels,n_frames))

    x1 = Reshape(target_shape=(size_y,size_x,n_channels*n_frames))(inp)
    x1 = GaussianNoise(0.05)(x1)
    x1 = Conv2D(64,(3,3),padding='same')(x1)
    x1 = SpatialDropout2D(0.5)(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    # x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D(pool_size=(2,2))(x1)

    x1 = GaussianNoise(0.03)(x1)
    x1 = Conv2D(128,(3,3),padding='same')(x1)
    x1 = SpatialDropout2D(0.5)(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    # x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D(pool_size=(2,2))(x1)

    x1 = GaussianNoise(0.02)(x1)
    x1 = Conv2D(128,(3,3),padding='same')(x1)
    x1 = SpatialDropout2D(0.5)(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    # x1 = BatchNormalization()(x1)

    # x1 = GaussianNoise(0.01)(x1)
    # x1 = Conv2D(256,(3,3),padding='same')(x1)
    # x1 = SpatialDropout2D(0.3)(x1)
    # x1 = LeakyReLU(alpha=0.2)(x1)
    # x1 = MaxPooling2D(pool_size=(2,2))(x1)

    x1 = Flatten()(x1)
    x1 = Dropout(0.3)(x1)
    x1 = Dense(units=h_units)(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)

    x1 = Dense(units=(size_y/4)*(size_x/4)*128)(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    x1 = Reshape((size_x/4,size_y/4,128))(x1)

    # x1 = UpSampling2D(size=(2,2))(x1)
    # x1 = Conv2D(256,(3,3),padding='same')(x1)
    # x1 = LeakyReLU(alpha=0.2)(x1)

    x1 = UpSampling2D(size=(2,2))(x1)
    x1 = Conv2D(128,(3,3),padding='same')(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    # x1 = BatchNormalization()(x1)

    x1 = UpSampling2D(size=(2,2))(x1)
    x1 = Conv2D(64,(3,3),padding='same')(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)



    x1 = Conv2D(n_frames,(3,3),activation='tanh', padding='same')(x1)
    x1 = Reshape(target_shape=(size_y,size_x,n_channels,n_frames))(x1)

    model = Model(inputs=[inp], outputs=[x1])
    rmsprop = RMSprop(lr=0.0001)

    model.compile(optimizer=rmsprop,loss='mse')

    return model

def small_2d_conv_net_dynamic(size_y=32,size_x=32,n_channels=1,n_frames=5,filters=10):

    inp = Input(shape=(size_y,size_x,n_channels,n_frames))
    x1 = Reshape(target_shape=(size_y,size_x,n_channels*n_frames))(inp)
    x1 = DynImage(filters=filters)(x1)
    x1 = GaussianNoise(0.05)(x1)
    x1 = Conv2D(32,(3,3),padding='same')(x1)
    x1 = SpatialDropout2D(0.3)(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    x1 = MaxPooling2D(pool_size=(2,2))(x1)

    x1 = GaussianNoise(0.02)(x1)
    x1 = Conv2D(64,(3,3),padding='same')(x1)
    x1 = SpatialDropout2D(0.3)(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    x1 = MaxPooling2D(pool_size=(2,2))(x1)

    x1 = GaussianNoise(0.01)(x1)
    x1 = Conv2D(256,(3,3),padding='same')(x1)
    x1 = SpatialDropout2D(0.3)(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    x1 = MaxPooling2D(pool_size=(2,2))(x1)


    # x1 = Flatten()(x1)
    # x1 = Dropout(0.3)(x1)
    # x1 = Dense(units=h_units)(x1)
    # x1 = LeakyReLU(alpha=0.2)(x1)
    #
    # x1 = Dense(units=(size_y/4)*(size_x/4)*128)(x1)
    # x1 = LeakyReLU(alpha=0.2)(x1)
    # x1 = Reshape((size_x/4,size_y/4,128))(x1)

    x1 = UpSampling2D(size=(2,2))(x1)
    x1 = Conv2D(256,(3,3),padding='same')(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)

    x1 = UpSampling2D(size=(2,2))(x1)
    x1 = Conv2D(128,(3,3),padding='same')(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)

    x1 = UpSampling2D(size=(2,2))(x1)
    x1 = Conv2D(64,(3,3),padding='same')(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)


    x1 = Conv2D(n_frames,(3,3),activation='tanh', padding='same')(x1)
    x1 = Reshape(target_shape=(size_y,size_x,n_channels,n_frames))(x1)

    model = Model(inputs=[inp], outputs=[x1])
    rmsprop = RMSprop(lr=0.0001)

    model.compile(optimizer=rmsprop,loss='mse')

    return model

def make_thresholds(ssim_feats, eta=0.3):

    mean = ssim_feats.mean(axis=0)
    mean = mean.reshape(1,mean.shape[0])

    cov = np.cov(ssim_feats, rowvar=False)
    cov_inv = np.linalg.inv(cov)

    distances = cdist(ssim_feats, mean, metric='mahalanobis',VI=cov_inv)

    sorted_distances = np.flip(np.sort(distances,axis=None), axis=0)
    farthest_distance_from_mean = sorted_distances[0]
    distance_eta = sorted_distances[int(len(sorted_distances) * eta)]

    return farthest_distance_from_mean, distance_eta, mean, cov_inv

def make_sorted_distances(ssim_feats_target,mean,cov_inv):

    distances = cdist(ssim_feats_target, mean, metric='mahalanobis',VI=cov_inv)

    sorted_distances = np.flip(np.sort(distances, axis=None), axis=0)

    return sorted_distances

def make_distance(feature,mean,cov_inv):

    distance = cdist(feature, mean, metric='mahalanobis', VI=cov_inv)

    return distance[0][0]



