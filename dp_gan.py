from __future__ import print_function

from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle
from PIL import Image

from six.moves import range

import tensorflow as tf
import keras.backend as K
from keras.datasets import mnist
from keras import layers
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
import numpy as np
import os
import argparse
import time

from privacy_accountant import accountant, utils
from custom_keras.noisy_optimizers import NoisyAdam

training_size = 6000
np.random.seed(1337)
K.set_image_data_format('channels_first')

target_eps = [0.125,0.25,0.5,1,2,4,8]
priv_accountant = accountant.GaussianMomentsAccountant(training_size)

def build_generator(latent_size):
    # we will map a pair of (z, L), where z is a latent vector and L is a
    # label drawn from P_c, to image space (..., 1, 28, 28)
    print('Generator')
    cnn = Sequential()

    cnn.add(Dense(256, input_dim=latent_size, activation='relu'))
    cnn.add(Dense(32 * 3 * 4, activation='relu'))
    cnn.add(Reshape((32, 3, 4)))

    # upsample to (..., 14, 14)
    cnn.add(UpSampling2D(size=(2, 6)))
    cnn.add(Conv2D(256, 5, padding='same',
                   activation='relu',
                   kernel_initializer='glorot_normal'))

    # take a channel axis reduction
    cnn.add(Conv2D(1, 4, strides=2, padding='same',
                   activation='linear',
                   kernel_initializer='glorot_normal'))

    # dense layer to reshape
    cnn.summary()

    # this is the z space commonly refered to in GAN papers
    latent = Input(shape=(latent_size, ))

    # this will be our label
    patient_class = Input(shape=(1,), dtype='int32')

    # 10 classes in MNIST
    cls = Flatten()(Embedding(
                        2, latent_size,
                        embeddings_initializer='glorot_normal')(patient_class))

    # hadamard product between z-space and a class conditional embedding
    h = layers.multiply([latent, cls])

    fake_patient = cnn(h)

    return Model([latent, patient_class], fake_patient)


def build_discriminator():
    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper
    print('Discriminator')
    cnn = Sequential()
    cnn.add(Conv2D(32, 3, padding='same', strides=2,
                   input_shape=(1, 3, 12)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(64, 3, padding='same', strides=1))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Flatten())
    cnn.add(Dense(1024, activation='relu'))
    cnn.add(Dropout(0.3))
    cnn.add(Dense(1024, activation='relu'))
    patient = Input(shape=(1, 3, 12))

    features = cnn(patient)
    cnn.summary()

    # first output (name=generation) is whether or not the discriminator
    # thinks the image that is being shown is fake, and the second output
    # (name=auxiliary) is the class that the discriminator thinks the image
    # belongs to.
    fake = Dense(1, activation='sigmoid', name='generation')(features)
    # aux could probably be 1 sigmoid too...
    aux = Dense(2, activation='softmax', name='auxiliary')(features)

    return Model(patient, [fake, aux])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", type=float, default=0)
    parser.add_argument("--clip_value", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--batch_size", type=int, default=100)

    args = parser.parse_args()

    print(args)
    epochs = args.epochs
    batch_size = args.batch_size
    latent_size = 100

    # Adam parameters suggested in https://arxiv.org/abs/1511.06434
    adam_lr = args.lr
    adam_beta_1 = 0.5

    directory = ('./output/' + str(args.noise) + '_' + str(args.clip_value) +
                 '_' + str(args.epochs) + '_' + str(args.lr) + '_' +
                 str(args.batch_size) + '/')

    if not os.path.exists(directory):
        os.mkdir(directory)

    if args.clip_value > 0:
        # build the discriminator
        discriminator = build_discriminator()
        discriminator.compile(
            optimizer=NoisyAdam(lr=adam_lr, beta_1=adam_beta_1,
                                clipnorm=args.clip_value,
                                noise=(args.noise * args.clip_value)),
            loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
        )
    else:
        discriminator = build_discriminator()
        discriminator.compile(
            optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
            loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
        )

    # build the generator
    generator = build_generator(latent_size)
    generator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
                      loss='binary_crossentropy')

    latent = Input(shape=(latent_size, ))
    image_class = Input(shape=(1,), dtype='int32')

    # get a fake image
    fake = generator([latent, image_class])

    # we only want to be able to train generation for the combined model
    discriminator.trainable = False
    fake, aux = discriminator(fake)
    combined = Model([latent, image_class], [fake, aux])

    combined.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )

    # get our mnist data, and force it to be of shape (..., 1, 28, 28) with
    # range [-1, 1]
    X_input = pickle.load(open('./data/X_processed.pkl', 'rb'))
    y_input = pickle.load(open('./data/y_processed.pkl', 'rb'))

    print(X_input.shape, y_input.shape)

    X_train = X_input[:training_size]
    X_test = X_input[training_size:]
    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)

    y_train = y_input[:training_size]
    y_test = y_input[training_size:]

    num_train, num_test = X_train.shape[0], X_test.shape[0]

    train_history = defaultdict(list)
    test_history = defaultdict(list)
    privacy_history = []

    with tf.Session() as sess:
        eps = tf.placeholder(tf.float32)
        delta = tf.placeholder(tf.float32)

        for epoch in range(epochs):
            print('Epoch {} of {}'.format(epoch + 1, epochs))

            num_batches = int(X_train.shape[0] / batch_size)
            progress_bar = Progbar(target=num_batches)

            epoch_gen_loss = []
            epoch_disc_loss = []

            train_start_time = time.clock()
            for index in range(num_batches):
                progress_bar.update(index)
                # generate a new batch of noise
                noise = np.random.uniform(-1, 1, (batch_size, latent_size))

                # get a batch of real images
                image_batch = X_train[index * batch_size:(index + 1) * batch_size]
                label_batch = y_train[index * batch_size:(index + 1) * batch_size]

                # sample some labels from p_c
                sampled_labels = np.random.randint(0, 2, batch_size)

                # generate a batch of fake images, using the generated labels as a
                # conditioner. We reshape the sampled labels to be
                # (batch_size, 1) so that we can feed them into the embedding
                # layer as a length one sequence
                generated_images = generator.predict(
                    [noise, sampled_labels.reshape((-1, 1))], verbose=0)

                # print(image_batch.shape)
                # print(generated_images.shape)
                X = np.concatenate((image_batch, generated_images))
                y = np.array([1] * batch_size + [0] * batch_size)
                aux_y = np.concatenate((label_batch, sampled_labels), axis=0)

                epoch_disc_loss.append(discriminator.train_on_batch(
                    X, [y, aux_y]))

                # make new noise. we generate 2 * batch size here such that we have
                # the generator optimize over an identical number of images as the
                # discriminator
                noise = np.random.uniform(-1, 1, (2 * batch_size, latent_size))
                sampled_labels = np.random.randint(0, 2, 2 * batch_size)

                # we want to train the generator to trick the discriminator
                # For the generator, we want all the {fake, not-fake} labels to say
                # not-fake
                trick = np.ones(2 * batch_size)

                epoch_gen_loss.append(combined.train_on_batch(
                    [noise, sampled_labels.reshape((-1, 1))],
                    [trick, sampled_labels]))
            print('\n Train time: ', time.clock() - train_start_time)
            print('accum privacy, batches: ' + str(num_batches))
            priv_start_time = time.clock()

            privacy_accum_op = priv_accountant.accumulate_privacy_spending(
                [None, None], args.noise, batch_size)
            for index in range(num_batches):
                with tf.control_dependencies([privacy_accum_op]):
                    spent_eps_deltas = priv_accountant.get_privacy_spent(
                        sess, target_eps=target_eps)
                    privacy_history.append(spent_eps_deltas)
                sess.run([privacy_accum_op])

            for spent_eps, spent_delta in spent_eps_deltas:
                print("spent privacy: eps %.4f delta %.5g" % (
                    spent_eps, spent_delta))
            print('priv time: ', time.clock() - priv_start_time)

            if spent_eps_deltas[-4][1] > 0.0001:
                raise Exception('spent privacy')

            print('\nTesting for epoch {}:'.format(epoch + 1))
            # generate a new batch of noise
            noise = np.random.uniform(-1, 1, (num_test, latent_size))

            # sample some labels from p_c and generate images from them
            sampled_labels = np.random.randint(0, 2, num_test)
            generated_images = generator.predict(
                [noise, sampled_labels.reshape((-1, 1))], verbose=False)

            print(sampled_labels[0])
            print(generated_images[0].astype(int))

            X = np.concatenate((X_test, generated_images))
            y = np.array([1] * num_test + [0] * num_test)
            aux_y = np.concatenate((y_test, sampled_labels), axis=0)

            # see if the discriminator can figure itself out...
            discriminator_test_loss = discriminator.evaluate(
                X, [y, aux_y], verbose=False)

            discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

            # make new noise
            noise = np.random.uniform(-1, 1, (2 * num_test, latent_size))
            sampled_labels = np.random.randint(0, 2, 2 * num_test)

            trick = np.ones(2 * num_test)

            generator_test_loss = combined.evaluate(
                [noise, sampled_labels.reshape((-1, 1))],
                [trick, sampled_labels], verbose=False)

            generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

            # generate an epoch report on performance
            train_history['generator'].append(generator_train_loss)
            train_history['discriminator'].append(discriminator_train_loss)

            test_history['generator'].append(generator_test_loss)
            test_history['discriminator'].append(discriminator_test_loss)

            print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
                'component', *discriminator.metrics_names))
            print('-' * 65)

            ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}'
            print(ROW_FMT.format('generator (train)',
                                 *train_history['generator'][-1]))
            print(ROW_FMT.format('generator (test)',
                                 *test_history['generator'][-1]))
            print(ROW_FMT.format('discriminator (train)',
                                 *train_history['discriminator'][-1]))
            print(ROW_FMT.format('discriminator (test)',
                                 *test_history['discriminator'][-1]))
            generator.save(
                directory +
                'params_generator_epoch_{0:03d}.h5'.format(epoch))

            if epoch > (epochs-10):
                discriminator.save(
                    directory +
                    'params_discriminator_epoch_{0:03d}.h5'.format(epoch))

            pickle.dump({'train': train_history, 'test': test_history,
                         'privacy': privacy_history},
                        open(directory + 'acgan-history.pkl', 'wb'))
