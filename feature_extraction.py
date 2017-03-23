# -*- coding: utf-8 -*-

import cv2
from functools import partial
import matplotlib.image as mpimg
from multiprocessing import Pool
import numpy as np
from os import cpu_count
from skimage.feature import hog

def extract_features(imgs,
                     color_space='RGB',
                     spatial_size=(32, 32),
                     hist_bins=32,
                     orient=9,
                     pix_per_cell=8,
                     cell_per_block=2,
                     hog_channel=0,
                     spatial_feat=True,
                     hist_feat=True,
                     hog_feat=True):
    """ Extract the features from a list of image files.

    Args:
        imgs: A list of the image file paths.
        color_space: The color space representation.
        spatial_size: The size of the image for spatial features.
        hist_bins: The number of bins for the histogram of the RGB channels.
        orient: The number of bins to divide the gradients into.
        pix_per_cell: The number of pixels per cell in calculating gradients.
        cell_per_block: The size of the block for normalizing the gradient
            histograms.
        hog_channel: The channels (0 = 'Red', 1 = 'Green', 2 = 'Blue',
            or 'All') to get HOG histograms.
        spatial_feat: A flag indicating whether to include spatial features.
        hist_feat: A flag indicating whether to include histograms of RGB
            channels.
        hog_feat: A flag indicating whether to include HOG features.
    Returns:
        A list of numpy arrays consisting of the features.
    """
    pool = Pool(processes=cpu_count())
    process_img = partial(single_img_features,
                          color_space=color_space,
                          spatial_size=spatial_size,
                          hist_bins=hist_bins,
                          orient=orient,
                          pix_per_cell=pix_per_cell,
                          cell_per_block=cell_per_block,
                          hog_channel=hog_channel,
                          spatial_feat=spatial_feat,
                          hist_feat=hist_feat,
                          hog_feat=hog_feat)
    features = pool.map(process_img, (img.as_posix() for img in imgs))
    return features

def single_img_features(image_file,
                        color_space='RGB',
                        spatial_size=(32, 32),
                        hist_bins=32,
                        orient=9,
                        pix_per_cell=8,
                        cell_per_block=2,
                        hog_channel=0,
                        spatial_feat=True,
                        hist_feat=True,
                        hog_feat=True):
    """ Extract the features from an image passed in as a numpy array.

    Args:
        image_file: Path to an image file.
        color_space: The color space representation.
        spatial_size: The size of the image for spatial features.
        hist_bins: The number of bins for the histogram of the RGB channels.
        orient: The number of bins to divide the gradients into.
        pix_per_cell: The number of pixels per cell in calculating gradients.
        cell_per_block: The size of the block for normalizing the gradient
            histograms.
        hog_channel: The channels (0 = 'Red', 1 = 'Green', 2 = 'Blue',
            or 'All') to get HOG histograms.
        spatial_feat: A flag indicating whether to include spatial features.
        hist_feat: A flag indicating whether to include histograms of RGB
            channels.
        hog_feat: A flag indicating whether to include HOG features.
    Returns:
        A numpy arrays consisting of the features.
    """
    image_features = []
    if 'png' in image_file:
        image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
    else:
        image = mpimg.imread(image_file)
    if spatial_feat:
        spatial_features = bin_spatial(image, color_space=color_space,
                                       size=spatial_size)
        image_features.append(spatial_features)
    if hist_feat:
        hist_features = color_hist(image, nbins=hist_bins)
        image_features.append(hist_features)
    if hog_feat:
        if hog_channel == 'ALL':
            hog_ft = []
            for channel in range(image.shape[2]):
                hog_ft.extend(hog_features(image[:, :, channel],
                                           orient=orient,
                                           pix_per_cell=pix_per_cell,
                                           cell_per_blk=cell_per_block,
                                           vis=False))
            hog_ft = np.ravel(hog_ft)
        else:
            hog_ft = hog_features(image[:, :, hog_channel],
                                  orient=orient,
                                  pix_per_cell=pix_per_cell,
                                  cell_per_blk=cell_per_block,
                                  vis=False)
        image_features.append(hog_ft)
    return np.concatenate(image_features)

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

def hog_features(img,
                 orient,
                 pix_per_cell,
                 cell_per_blk,
                 vis=False,
                 feature_vector=True):
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
        The HOG histograms as a numpy array.
    """
    if vis:
        features, hog_img = hog(img,
                                orientations=orient,
                                pixels_per_cell=(pix_per_cell, pix_per_cell),
                                cells_per_block=(cell_per_blk, cell_per_blk),
                                visualise=vis,
                                feature_vector=feature_vector)
        return features, hog_img
    else:
        features = hog(img,
                       orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_blk, cell_per_blk),
                       visualise=vis,
                       feature_vector=feature_vector)
        return features