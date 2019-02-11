'''
The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to  generate latent vectors.
The decoder can be used to generate font images by sampling the
latent vector from a Gaussian distribution with mean=0 and std=1.

# Reference
Kingma, Diederik P., and Max Welling.
"Auto-encoding variational bayes."
https://arxiv.org/abs/1312.6114
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten, Lambda, Dropout
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard, LambdaCallback, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.backend import set_session

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import datetime
from os import makedirs
from os.path import exists, join

config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

#################################################################################################
#################################################################################################
def plot_results(models, data, batch_size=128, model_name="font_vae"):

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)
    now = datetime.datetime.now()

    # y_label = validation_generator.classes
    # class_dictionary = validation_generator.class_indices

    name = "plot/font_vae_mean_"+now.strftime("%m %d %H %M %S")+".png"
    filename = os.path.join(model_name, name)

    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test, batch_size=batch_size)
    # z_mean, _, _ = encoder.predict_generator(x_test, steps=len(validation_generator))

    n_class = y_label.max() + 1

    plt.figure(figsize=(18, 15))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_label,
                cmap=plt.cm.get_cmap('tab10', n_class), s=5, alpha=0.5) # tab10, gist_rainbow
    font_name_list = list(plot_generator.class_indices.keys())
    # ['7_DancingScript-Regular', '5_Bangers-Regular', '4_Righteous-Regular',
    # '6_Pacifico-Regular','8_Inconsolata-Regular', '1_PT_Serif-Web-Regular',
    # '0_EBGaramond-Regular', '3_Roboto-Regular','2_NotoSans-Regular', '9_VT323-Regular']

    cbar = plt.colorbar(ticks=range(n_class))
    # cbar.set_ticklabels(font_name_list)
    plt.xlabel("z_mean[0]")
    plt.ylabel("z_mean[1]")
    plt.savefig(filename)
    plt.close('all')
    # plt.show()

#################################################################################################
#################################################################################################
# Generating dataset
batch_size = 128
image_size = 112
train_datagen = ImageDataGenerator(
    # rotation_range=20,
    # zoom_range=0.2,
    # horizontal_flip=True,
    rescale=1./255
)

test_datagen = ImageDataGenerator(rescale=1./255)
plot_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/dataset/train',
    target_size=(112, 112),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='input'
)

validation_generator = test_datagen.flow_from_directory(
    'data/dataset/validation',
    target_size=(112, 112),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='input'
    # shuffle=False
)

plot_generator = plot_datagen.flow_from_directory(
    # 'data/valdataset',
    'data/dataset/validation',
    target_size=(112, 112),
    batch_size=validation_generator.n,
    color_mode='grayscale',
    class_mode='input',
    shuffle=False
)


# _x_test, _y_test = zip(*(validation_generator[i] for i in range(len(validation_generator))))
# x_test = np.vstack(_x_test)
# y_test = np.vstack(_y_test)
# y_label = validation_generator.index_array
x_test, y_test = next(validation_generator)
x_plot, y_plot = next(plot_generator)
y_label = plot_generator.classes
# class_dictionary = validation_generator.class_indices


# network parameters
input_shape = (image_size, image_size, 1)
kernel_size = 3
filters = 16
latent_dim = 2
epochs = 2
log_dir='./logs'

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='Encoder_input')
x = inputs
# for i in range(4):
#     filters *= 2
#     x = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', strides=2, padding='same')(x)

x = Conv2D(filters=32, kernel_size=kernel_size, activation='relu', strides=2, padding='same')(x)
x = Dropout(0.2)(x)
x = Conv2D(filters=64, kernel_size=kernel_size, activation='relu', strides=2, padding='same')(x)
x = Dropout(0.2)(x)
x = Conv2D(filters=128, kernel_size=kernel_size, activation='relu', strides=2, padding='same')(x)
x = Dropout(0.2)(x)
x = Conv2D(filters=256, kernel_size=kernel_size, activation='relu', strides=2, padding='same')(x)

# shape info needed to build decoder model
shape = K.int_shape(x)

# generate latent vector Q(z|X)
x = Flatten()(x)
x = Dense(16, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='Encoder')
encoder.summary()
plot_model(encoder, to_file='summary/font_vae_cnn_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

# for i in range(4):
#     x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, activation='relu', strides=2, padding='same')(x)
#     filters //= 2

x = Conv2DTranspose(filters=256, kernel_size=kernel_size, activation='relu', strides=2, padding='same')(x)
x = Dropout(0.2)(x)
x = Conv2DTranspose(filters=128, kernel_size=kernel_size, activation='relu', strides=2, padding='same')(x)
x = Dropout(0.2)(x)
x = Conv2DTranspose(filters=64, kernel_size=kernel_size, activation='relu', strides=2, padding='same')(x)
x = Dropout(0.2)(x)
x = Conv2DTranspose(filters=32, kernel_size=kernel_size, activation='relu', strides=2, padding='same')(x)

outputs = Conv2DTranspose(filters=1, kernel_size=kernel_size, activation='sigmoid', padding='same', name='Decoder_output')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='Decoder')
decoder.summary()
plot_model(decoder, to_file='summary/font_vae_cnn_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='VAE')

#################################################################################################
#################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m", "--mse", help=help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)
    data = (x_test, y_test)
    plot_data = (x_plot, y_plot)


    def vae_loss_custom(y_true, y_pred):
        # xent_loss = image_size * image_size * binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
        xent_loss = binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
        # xent_loss = image_size * image_size * mse(K.flatten(y_true), K.flatten(y_pred))
        # kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        kl_loss = -5e-4 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        vae_loss = K.mean(xent_loss + kl_loss)
        return vae_loss

    vae.compile(optimizer='rmsprop', loss=vae_loss_custom, metrics=['accuracy'])
    # vae.compile(optimizer='adadelta', loss=vae_loss_custom, metrics=['accuracy'])
    vae.summary()
    plot_model(vae, to_file='summary/font_vae_cnn.png', show_shapes=True)

    tb_hist = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
    # checkpoint = ModelCheckpoint(filepath='font_vae_cnn/model/checkpoint-{epoch:02d}.hdf5')
    lcb = LambdaCallback(on_epoch_end=lambda epoch, logs: plot_results(models, plot_data, batch_size=batch_size, model_name="font_vae_cnn"))
    # checkpoint_model = LambdaCallback(on_epoch_end=lambda epoch, logs: vae.save('font_vae_cnn/model/mymodel.h5'))

    hist = vae.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[lcb, tb_hist]
    )
    # vae.save_weights('font_vae_cnn.h5')

    # plot_results(models, data, batch_size=batch_size, model_name="font_vae_cnn")
    # plot_results(models, plot_data, batch_size=batch_size, model_name="font_vae_cnn")

    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='upper left')

    acc_ax.plot(hist.history['acc'], 'b', label='train acc')
    acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
    acc_ax.set_ylabel('accuracy')
    acc_ax.legend(loc='lower left')
    plt.savefig("font_vae_cnn/history.png")
