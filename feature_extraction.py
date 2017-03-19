# -*- coding: utf-8 -*-

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog

__author__ = 'Johnson Jia'

def color_hist(img, nbins=32, bins_range=(0, 256)):
    """ Compute the histogram of the RGB channels separately.

    Args:
        img: The image in RBG space as an numpy array.
        nbins: The number of bins in the histogram.
        bins_range: The range ot the bins.

    Returns:
        The histogram of color intensities as a feature vector.
    """
    rhist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    ghist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    bhist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    return hist_features

def bin_spatial(img, color_space='RGB', size=(32, 32)):
    """ Compute the color histogram features.

    Args:
        img: The image in RGB space as an numpy array.
        color_space: The color space to use, can be one of 'RGB', 'HSV',
            'LUV', 'HLS', 'YUV', 'YCrCb'.
        size: The output image size.

    Returns:
        The spatial space istogram as a feature vector.
    """
    if color_space == 'RGB':
        feature_image = np.copy(img)
    else:
        feature_image = cv2.cvtColor(img, eval('cv2.COLOR_RGB2{cs}'.
                                               format(cs=color_space)))
    features = cv2.resize(feature_image, size).ravel()
    return features

def hof_features(img,
                 orient,
                 pix_per_cell,
                 cell_per_blk,
                 vis=False):
    """ Calculate Histogram of Oriented Gradients (HOG) as a feature vector.

    Args:
        img: The image in RGB space as an numpy array.
        orient: The number of orientation bins to split up the
            gradient information.
        pix_per_cell: The number of pixels per cell over which each gradient
            histogram is computed.
        cell_per_block: The size of the block over which the histogram counts
            in a given cell will be normalized.
        vis: A flag to specify whether to return the HOG image or not.

    Returns:

    """
    if vis:
        features, hog_img = hog(img,
                                orientations=orient,
                                pixels_per_cell=(pix_per_cell, pix_per_cell),
                                cells_per_block=(cell_per_blk, cell_per_blk),
                                visualise=True,
                                feature_vector=True)
        return features, hog_img
    else:
        features = hog(img,
                       orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_blk, cell_per_blk),
                       visualise=True,
                       feature_vector=True)
        return features