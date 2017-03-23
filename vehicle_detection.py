# -*- coding: utf-8 -*-

import click
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle

from feature_extraction import bin_spatial, color_hist, hog_features
from train_model import train_model

@click.command()
@click.option('--model_pickle', default='test.pkl',
              help='the pickle file of the trained model')
@click.option('--video', help='video file')
@click.option('--image', default='test_images/test1.jpg',
              help='pickled features data')
@click.argument('output_file')
def vehicle_detection(model_pickle, video, image, output_file):
    """

    Args:
        model_pickle:
        video:
        image:
        output_video:

    Returns:

    """
    if video == None:
        model_pickle = pickle.load(open(model_pickle, 'rb'))
        clf = model_pickle['clf']
        X_scaler = model_pickle['scaler']
        img = mpimg.imread(image)
        windows = []
        for s in [0.8, 0.9, 1, 1.2, 1.5, 1.8, 2, 2.5]:
            windows.extend(find_cars(img, clf=clf, X_scaler=X_scaler,
                                     x_start_stop=(0, 1280),
                                     y_start_stop=(400, 656),
                                     scale=s))
        for window in windows:
            cv2.rectangle(img, window[0], window[1], (0, 0, 255), 6)
        plt.imshow(img)
        plt.show()

def find_cars(img,
              clf,
              X_scaler,
              x_start_stop=(None, None),
              y_start_stop=(None, None),
              color_space='RGB',
              scale=1,
              orient=9,
              pix_per_cell=8,
              cell_per_block=2,
              spatial_size=(32, 32),
              hist_bins=16):
    """ Find the squares around cars in the image.

    Args:
        img: The input image as a numpy array.
        clf: The model used to identify cars.
        X_scaler: The feature scaling function.
        x_start_stop: The start-stop coordinates in the x-direction.
        y_start_stop: The start-stop coordinates in the y-direction.
        color_space: The color space to use for color histogram.
        scale: The scaling factor on the image.
        orient: The number of orientations to apply HOG.
        pix_per_cell: The number of pixels per cell.
        cell_per_block: The number of cells per block used in HOG.
        spatial_size: The size to down-sample the image for spatial binning.
        hist_bins: The number of bins to divide color spaces.

    Returns:
        A list of coordinates of boxes enclosing the parts of the image
        where the model identified as images of cars.
    """
    windows = []
    img = img.astype(np.float32) / 255

    img_tosearch = img[y_start_stop[0]:y_start_stop[1],
                       x_start_stop[0]:x_start_stop[1], :]
    ctrans_tosearch = img_tosearch
    if scale != 1:
        imshape = img_tosearch.shape
        ctrans_tosearch = cv2.resize(img_tosearch,
                                     (np.int(imshape[1]/scale),
                                      np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - 1
    nfeat_per_block = orient * cell_per_block ** 2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = hog_features(ch1, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vector=False)
    hog2 = hog_features(ch2, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vector=False)
    hog3 = hog_features(ch3, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vector=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window,
                             xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window,
                             xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window,
                             xpos:xpos + nblocks_per_window].ravel()
            hog_feats = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window,
                                                xleft:xleft + window],
                                (64, 64))
            # Get color features
            spatial_features = bin_spatial(subimg, color_space=color_space,
                                           size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack(
                (spatial_features, hist_features, hog_feats)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = clf.predict(test_features)

            if test_prediction == 1:
                xbox_l = np.int(xleft * scale)
                y_top = np.int(ytop * scale)
                xbox_r = xbox_l + np.int(window * scale)
                y_bottom = y_top + np.int(window * scale)
                windows.append(((xbox_l+x_start_stop[0],
                                 y_top+y_start_stop[0]),
                                (xbox_r+x_start_stop[0],
                                 y_bottom+y_start_stop[0])))
    return windows

if __name__ == '__main__':
    vehicle_detection()