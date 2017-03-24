# -*- coding: utf-8 -*-

import click
import cv2
from collections import deque
from functools import partial
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import numpy as np
import pickle
from scipy.ndimage.measurements import label

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
    model_pickle = pickle.load(open(model_pickle, 'rb'))
    clf = model_pickle['clf']
    X_scaler = model_pickle['scaler']
    # Identify vehicle in an image.
    if video == None:
        img = mpimg.imread(image)
        windows = []
        for s in [0.8, 0.9, 1, 1.2, 1.5, 1.8]:
            windows.extend(find_cars(img, clf=clf, X_scaler=X_scaler,
                                     x_start_stop=(0, 1280),
                                     y_start_stop=(400, 656),
                                     scale=s))
        img_boxes = draw_boxes(img, windows)
        plt.imshow(img_boxes)
        plt.show()
    # Track vehicle in a video.
    else:
        clip = VideoFileClip(video)
        heatmap = Heatmap(n_frame_avg_over=5, threshold=10)
        func = partial(process_frame, clf=clf,
                       X_scaler=X_scaler, heatmap=heatmap)
        clip_w_boxes = clip.fl_image(func)
        clip_w_boxes.write_videofile(output_file, audio=False)

def process_frame(img,
                  clf,
                  X_scaler,
                  heatmap):
    """ A helper function used to detect cars over multiple frames in a video.

    Args:
        img: The frame image.
        clf: The model for identifying vehicles.
        X_scaler: The feature scaler.

    Returns:
        The image with cars drawn.
    """
    windows = []
    for s in [0.8, 1, 1.2, 1.5, 2, 2.5]:
        windows.extend(find_cars(img, clf=clf, X_scaler=X_scaler,
                                 x_start_stop=(0, 1280),
                                 y_start_stop=(400, 656),
                                 scale=s))
    img_boxes = heatmap.draw_label_boxes(img, windows)
    return img_boxes

class Heatmap:

    def __init__(self, n_frame_avg_over=5, threshold=1):
        self._n_frame_avg_over = n_frame_avg_over
        self._threshold = threshold
        self._frames = []
        self._windows_queue = deque(maxlen=n_frame_avg_over)

    def draw_label_boxes(self, img, windows):
        """ Draw boxes over labeled regions.

        Args:
            img: The original image.
            windows: The collection of windows identified.

        Returns:
            The image with boxes drawn around cars.
        """
        self._windows_queue.append(windows)
        self.heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
        for bbox_list in self._windows_queue:
            for box in bbox_list:
                self.heatmap[box[0][1]:box[1][1],
                             box[0][0]:box[1][0]] += 1
        # Apply threshold
        self.heatmap[self.heatmap < self._threshold] = 0
        labels = label(self.heatmap)

        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                    (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        # Return the image
        return img

def draw_boxes(img, windows):
    """ Draw windows on the image.

    Args:
        img: The base image.
        windows: Coordinates on the image where boxes are.

    Returns:
        The image with boxes drawn.
    """
    image = np.copy(img)
    for window in windows:
        cv2.rectangle(image, window[0], window[1], (0, 0, 255), 6)
    return image

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