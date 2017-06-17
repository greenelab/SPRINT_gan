import os
import argparse
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle as pkl
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from glob import glob

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

acgan = __import__('ac_gan')
from keras.models import load_model


def run(id):
    latent_size = 100

    lr_clf = linear_model.LogisticRegression()
    transfer_clf = RandomForestClassifier()

    mean_scores = []
    lr_scores = []
    directory = './output/p' + str(id) + '_8.0_0.0001_500_0.002_100/'
    # directory = './output/acgan_500_0.0002_100/'

    for i in range(0, 500):
        gen_name = sorted(glob(directory + 'params_generator*'))[i]
        print(gen_name)
        g = load_model(gen_name)

        generate_count = training_size

        noise = np.random.uniform(-1, 1, (generate_count, latent_size))
        sampled_labels = np.random.randint(0, 2, generate_count)
        generated_images = g.predict([noise, sampled_labels.reshape((-1, 1))],
                                     verbose=0)

        gen_X_train = np.reshape(generated_images, (training_size, 3, 12))
        gen_X_train = gen_X_train.astype(int)
        gen_X_train = gen_X_train.clip(min=0)

        gen_X_train = gen_X_train.reshape(generate_count, -1)
        gen_y_train = sampled_labels


        mean_scores.append(accuracy_score(y_test, transfer_clf.fit(gen_X_train, gen_y_train).predict(X_test)))
        lr_scores.append(accuracy_score(y_test, lr_clf.fit(gen_X_train, gen_y_train).predict(X_test)))
    pkl.dump({'rf': mean_scores, 'lr': lr_scores}, open(directory + 'epoch_scores.p', 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int)

    args = parser.parse_args()
    if args.id is not None:
        run(args.id)
