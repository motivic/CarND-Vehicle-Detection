# -*- coding: utf-8 -*-

import cv2
import click
import numpy as np
import pathlib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import time

from feature_extraction import extract_features

COLOR_SPACE = 'RGB'
ORIENT = 9
PIX_PER_CELL = 8
CELL_PER_BLOCK = 2
HOG_CHANNEL = 'ALL'
SPATIAL_SIZE = (32, 32)
HIST_BINS = 16
SPATIAL_FEAT = True
HIST_FEAT = True
HOG_FEAT = True
Y_START_STOP = [480, 720]

@click.command()
@click.option('--model_type', default='SVC',
              help='the type of model to use')
@click.option('--car', default='cars',
              help='folder containing car images')
@click.option('--noncar', default='non-cars',
              help='folder containing non-car images')
@click.option('--load_pickles', default=0,
              help='1 to load pickles, 0 to regenerate the pickles')
@click.option('--features_pickle', default='features.pkl',
              help='pickled features data')
@click.option('--labels_pickle', default='label.pkl',
              help='pickled features data')
@click.option('--video', default='test_video.mp4',
              help='pickled features data')
@click.argument('output_file')
def train_model(model_type, car, noncar, load_pickles,
                features_pickle, labels_pickle, video, output_file):
    if load_pickles == 1:
        with open(features_pickle, 'rb') as infile:
            X = pickle.load(infile)
        with open(labels_pickle, 'rb') as infile:
            y = pickle.load(infile)
    else:
        # Read in the image file paths
        p = pathlib.Path(car)
        car_imgs = list(p.glob('**/*.png')) + \
                   list(p.glob('**/*.jpg')) + \
                   list(p.glob('**/*.jpeg'))
        p = pathlib.Path(noncar)
        noncar_imgs = list(p.glob('**/*.png')) + \
                      list(p.glob('**/*.jpg')) + \
                      list(p.glob('**/*.jpeg'))

        car_features = extract_features(car_imgs,
                                        color_space=COLOR_SPACE,
                                        spatial_size=SPATIAL_SIZE,
                                        hist_bins=HIST_BINS,
                                        orient=ORIENT,
                                        pix_per_cell=PIX_PER_CELL,
                                        cell_per_block=CELL_PER_BLOCK,
                                        hog_channel=HOG_CHANNEL,
                                        spatial_feat=SPATIAL_FEAT,
                                        hist_feat=HIST_FEAT,
                                        hog_feat=HOG_FEAT)
        noncar_features = extract_features(noncar_imgs,
                                           color_space=COLOR_SPACE,
                                           spatial_size=SPATIAL_SIZE,
                                           hist_bins=HIST_BINS,
                                           orient=ORIENT,
                                           pix_per_cell=PIX_PER_CELL,
                                           cell_per_block=CELL_PER_BLOCK,
                                           hog_channel=HOG_CHANNEL,
                                           spatial_feat=SPATIAL_FEAT,
                                           hist_feat=HIST_FEAT,
                                           hog_feat=HOG_FEAT)
        X = np.vstack((car_features, noncar_features)).astype(np.float64)
        with open(features_pickle, 'wb') as outfile:
            pickle.dump(X, outfile)
        y = np.hstack((np.ones(len(car_features)),
                       np.zeros(len(noncar_features))))
        with open(labels_pickle, 'wb') as outfile:
            pickle.dump(y, outfile)

    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y,
                                                        test_size=0.2,
                                                        random_state=rand_state,
                                                        stratify=y)
    # Train the model
    model = None
    if model_type == 'SVC':
        model = LinearSVC()
        t = time.time()
        model.fit(X_train, y_train)
        t2 = time.time()
        print('Training SCV took {:.2f} seconds...'.format(t2-t))
        print('Test accuracy of SCV is {:.4f}.'.format(model.score(X_test,
                                                                   y_test)))
        with open(output_file, 'wb') as outfile:
            pickle.dump(model, output_file)
    return model

if __name__ == '__main__':
    train_model()