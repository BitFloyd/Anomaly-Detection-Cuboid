import matplotlib as mpl
mpl.use('Agg')
from keras import backend
from keras.layers import Input, Dense, Lambda,Reshape,GaussianNoise,Conv2D,SpatialDropout2D,LeakyReLU,MaxPooling2D
from keras.layers import UpSampling2D,Flatten
from keras.models import Model
from keras.objectives import mean_squared_error as mse
from keras.optimizers import Adam
# from keras.utils import vis_utils as vizu
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import socket
from functionals_pkg.logging import Logger

if(socket.gethostname()=='puck'):

    print "############################################"
    print "DETECTED RUN ON PUCK"
    print "############################################"

    path_videos = '/usr/local/data/sejacob/ANOMALY/data/UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test'
    train_test = 'Test'
    verbose = 1

elif('gpu' in socket.gethostname()):
    print "############################################"
    print "DETECTED RUN ON HELIOS: Probably"
    print "############################################"
    verbose = 1
    os.chdir('/scratch/suu-621-aa/ANOMALY/cuboid')
    path_videos = '/scratch/suu-621-aa/ANOMALY/data/UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test'
    train_test = 'Test'
    sys.stdout=Logger(key = os.path.join('Results','train_vae'))

else:
    print socket.gethostname()
    print "############################################"
    print "DETECTED RUN ON GUILLIMIN: Probably"
    print "############################################"
    verbose = 1
    os.chdir('/gs/project/suu-621-aa/sejacob/cuboid/')
    sys.stdout=Logger(key = os.path.join('Results','train_vae'))


intermediate_dim = 256
latent_dim = 32
batch_size = 98
nb_epoch = 100
epsilon_std = .01

size_x=12
size_y=12
n_channels=1
n_frames=5

#Encoder
inp = Input(shape=(size_y, size_x, n_channels, n_frames))

x1 = Reshape(target_shape=(size_y, size_x, n_channels * n_frames))(inp)
x1 = GaussianNoise(0.05)(x1)
x1 = Conv2D(32, (3, 3), padding='same')(x1)
x1 = SpatialDropout2D(0.5)(x1)
x1 = LeakyReLU(alpha=0.2)(x1)
x1 = MaxPooling2D(pool_size=(2, 2))(x1)

x1 = GaussianNoise(0.03)(x1)
x1 = Conv2D(64, (3, 3), padding='same')(x1)
x1 = SpatialDropout2D(0.5)(x1)
x1 = LeakyReLU(alpha=0.2)(x1)
x1 = MaxPooling2D(pool_size=(2, 2))(x1)

x1 = GaussianNoise(0.02)(x1)
x1 = Conv2D(64, (3, 3), padding='same')(x1)
x1 = SpatialDropout2D(0.5)(x1)
x1 = LeakyReLU(alpha=0.2)(x1)
x1 = Flatten()(x1)
x1 = Dense(units=intermediate_dim)(x1)
hidden = LeakyReLU(alpha=0.2)(x1)

z_mean = Dense(units=latent_dim)(hidden)
z_log_var = Dense(units=latent_dim)(hidden)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = backend.random_normal(shape=(backend.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + backend.exp(z_log_var) * epsilon


z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

#Decoder
dec0 = Dense(units=intermediate_dim)
dec00 = LeakyReLU(alpha=0.2)
dec1 = Dense(units=(64*9))
dec2 = LeakyReLU(alpha=0.2)
dec3 = Reshape(target_shape=(3,3,64))
dec4 = UpSampling2D(size=(2, 2))
dec5 = Conv2D(128, (3, 3), padding='same')
dec6 = LeakyReLU(alpha=0.2)


dec7 = UpSampling2D(size=(2, 2))
dec8 = Conv2D(64, (3, 3), padding='same')
dec9 = LeakyReLU(alpha=0.2)

dec10 = Conv2D(n_frames, (3, 3), activation='tanh', padding='same')
dec11 = Reshape(target_shape=(size_y, size_x, n_channels, n_frames))

dec0_layer = dec0(z)
dec00_layer = dec00(dec0_layer)
dec1_layer = dec1(dec00_layer)
dec2_layer = dec2(dec1_layer)
dec3_layer = dec3(dec2_layer)
dec4_layer = dec4(dec3_layer)
dec5_layer = dec5(dec4_layer)
dec6_layer = dec6(dec5_layer)
dec7_layer = dec7(dec6_layer)
dec8_layer = dec8(dec7_layer)
dec9_layer = dec9(dec8_layer)
dec10_layer = dec10(dec9_layer)
x_decoded = dec11(dec10_layer)


vae = Model(inputs=[inp], outputs=[x_decoded])

# vizu.plot_model(vae, "ff.png", show_layer_names=False, show_shapes=True)


def vae_objective(x, x_decoded):
    loss = mse(x, x_decoded)
    kl_regu = - 0.5 * backend.mean(1 + z_log_var - backend.square(z_mean) - backend.exp(z_log_var), axis=-1)
    return loss + kl_regu

adam = Adam(lr=0.000001)

vae.compile(optimizer=adam, loss=vae_objective)
vae.summary()

encoder = Model(inputs=[inp], outputs=[z_mean])
# vizu.plot_model(encoder, "vae_encoder.png", show_layer_names=False, show_shapes=True)

decoder_input = Input(shape=(latent_dim,))
dec0_layer_g = dec0(decoder_input)
dec00_layer_g = dec00(dec0_layer_g)
dec1_layer_g = dec1(dec00_layer_g)
dec2_layer_g = dec2(dec1_layer_g)
dec3_layer_g = dec3(dec2_layer_g)
dec4_layer_g = dec4(dec3_layer_g)
dec5_layer_g = dec5(dec4_layer_g)
dec6_layer_g = dec6(dec5_layer_g)
dec7_layer_g = dec7(dec6_layer_g)
dec8_layer_g = dec8(dec7_layer_g)
dec9_layer_g = dec9(dec8_layer_g)
dec10_layer_g = dec10(dec9_layer_g)
x_decoded_g = dec11(dec10_layer_g)

generator = Model(inputs=[decoder_input], outputs=[x_decoded_g])

# vizu.plot_model(generator, "vae_generator.png", show_layer_names=False, show_shapes=True)

print "LOADING CUBOIDS AND FEATS"
all_cuboids_normed = np.load(os.path.join('data_stored_temporal', 'all_cuboids_normed.npy'))


weights_file = "vae_%d_latent.hdf5" % latent_dim
if os.path.isfile(os.path.join('saved_models',weights_file)):
    print "LOADING WEIGHTS FILE:"
    vae.load_weights(os.path.join('saved_models',weights_file))
else:
    from keras.callbacks import History

    hist_cb = History()
    vae.fit(all_cuboids_normed, all_cuboids_normed, shuffle=True, epochs=nb_epoch, batch_size=batch_size,
            callbacks=[hist_cb],verbose=1)
    vae.save_weights(os.path.join('saved_models',weights_file))

    # plot convergence curves to show off
    plt.plot(hist_cb.history["loss"], label="training")
    plt.plot(hist_cb.history["val_loss"], label="validation")
    plt.grid("on")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc="best")
    plt.savefig(os.path.join('graphs','vae_train_plot'))