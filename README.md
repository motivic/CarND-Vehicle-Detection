# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

##Writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1a]: ./output_images/vehicle.png
[image1b]: ./output_images/non-vehicle.png
[image2]: ./output_images/bin_spatial.jpg
[image4a]: ./output_images/out_test1.jpg
[image4b]: ./output_images/out_test2.jpg
[image4c]: ./output_images/out_test3.jpg
[image4d]: ./output_images/out_test4.jpg
[image4e]: ./output_images/out_test5.jpg
[image4f]: ./output_images/out_test6.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is in the file `feature_extraction.py`, [lines 156-192](https://github.com/motivic/CarND-Vehicle-Detection/blob/master/feature_extraction.py#L156-L191).
The function `hog_features` accepts an input image and HOG parameters (`orientation`, `pix_per_cell`, and `cell_per_blk`),
then calls the `hog` function from OpenCV and returns the HOG features.

For training the model, I started by reading in all the `vehicle`, `vehicle_smallest`, `non-vehicle`, and `non-vehicle_smallest`
images. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1a]
![alt text][image1b]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). 
I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

From these explorations, I decided to try to use the YCrCb color space for both the spatial bins and for generating HOG features (just channel 0).

####2. Explain how you settled on your final choice of HOG parameters.

I started with the parameters used in the course, `orientation=9`, `pixels_per_cell=(8, 8)`, and `cells_per_block=(2, 2)`, to extract HOG
features, combine with color histogram and spatial bin features, train a linear SVM model, and evaluate on the test images and test video. The model
trained with these parameters happened to work quite well. So I decided to not tweak the HOG parameters, as they seem to be informative features, but
instead focused on training a good model on these features, which we explain in the following section.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I used color histograms on the RGB colorspace, spatial bins on the YGrGb space, and HOG on all three channels of the YGrGb space as the features.

I started with a simple linear SVM (`train_model.py`, [lines 125-127](https://github.com/motivic/CarND-Vehicle-Detection/blob/master/train_model.py#L126-127))
and got over 99% accuracy when evaluated on the hold-out test set. When using the model to identify cars in the
test images, there are still some mis-identifications. So I tried to train two more complex models using hyperparameter search. One of these is a 
grid search on the parameter space of a SVM. This again gave a test set accuracy over 99%, but it turned out to be impractical as using the model 
to process the `project_video.mp4` showed a progress estimate of around 36 hours.

The other more complex model I trained is a Gradient Boosted Tree model with randomized hyperparameter search. This is also in the file `train_model.py`,
[lines 128-133](https://github.com/motivic/CarND-Vehicle-Detection/blob/master/train_model.py#L121-L128). This model took quite a while (around 1 hour) to 
train even when I had used all 8 cores on my machine to train it. This model also had good accuracy (over 99%) on the test set. However, processing 
`project_video.mp4` using this GBM model took over 6 hours, and the result is comparable to that by the simple linear SVM. So in the end I chose the linear
SVM as the model of choice.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I applied sliding window search to detect vehicles, but for a limited range of pixels with y-coordinate between 400 and 656 (`vehicle_detection.py`, 
[35-36](https://github.com/motivic/CarND-Vehicle-Detection/blob/master/vehicle_detection.py#L35-L36)). The implementation of the sliding window and feature
generation is in the function `find_cars` in `vehicle_detection.py`, [lines 156-255](https://github.com/motivic/CarND-Vehicle-Detection/blob/master/vehicle_detection.py#L156-L255).
We followed essentially the HOG sub-sampling window search from the course, but using only one color channel for the HOG features. 

Because the featuers need to be of the same size, instead of scaling the windows, I scaled the base image (or video frame) by (1/(0.8), 1, 1/(1.5)),
and applied the window search algorithm. These scales are chosen by trial and error. See lines [71](https://github.com/motivic/CarND-Vehicle-Detection/blob/master/vehicle_detection.py#L71) 
and [191-194](https://github.com/motivic/CarND-Vehicle-Detection/blob/master/vehicle_detection.py#L191-L194) of `vehicle_detection.py`.

Instead of defining an overlap, we followed the HOG sub-sampling window search from the course and specified the number of cells to step over
from each window to the next. We left this parameter at 2, and given that each cell has 8 pixels, this means the neighboring windows are 16 pixels apart
(after scaling). See [line 205](https://github.com/motivic/CarND-Vehicle-Detection/blob/master/vehicle_detection.py#L205) of `vehicle_detection.py`


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4a] ![alt text][image4b]
![alt text][image4c] ![alt text][image4d]
![alt text][image4e] ![alt text][image4f]

We see there are some mis-identifications. The model has misclassified sections of the left yellow lane line as parts of vehicles in the second and fourth images, 
and one can argue whether the left-most box in the first image is a valid identification or not (there is indeed a vehicle in the box, but it is
occluded by the center divider/median strip). Nevertheless, I'm satisfied with the performance of the model as the windows around the vehicles are correctly boxed (except 
for the fifth image, where part of the white car are not boxed).

We can re-train the model to address the specific mis-identifications by including these images as a part of the "non-car" training data. Also
for the false negatives (where the window is over a part of a car but it did not get boxed), we can add these images into the "car" training set
so they are more likely to be boxed after retraining the model. For now, we will continue to use the same model for detecting vehicles in the video.

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The filtering I used follows the heatmap method from the course. I recorded the positions of the positive detections in each frame of the video. 
I then summed the positive detections over a few frames (three is what I used in the end) and created a heatmap. Then I thresholded that heatmap 
to identify vehicle positions. Next I used the function `scipy.ndimage.measurements.label` to identify individual blobs in the heatmap, with
 each blob assumed to correspond to a vehicle. Finally I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label` and 
the bounding boxes then overlaid on the last frame of video:

### Here are six frames showing the original image, the heatmap, the vehicle labels, and the image with bounding boxes drawn around the vehicles. 

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The approach I took for the vehicle detection project closely follows that suggested in the course. The details have been discussed in the sections above.

The main issue I faced during the project is in reduing (or eliminating) erroneous detections. For this, I tried adjusting the color space, training a more accurate model as well as adjusting the
threshold applied to the heatmap and the number of frames to aggregate over. Training a more complex model did not improve the error rate and only made processing the
videos a lot slower. Adjusting the color space and the feature generation did not improve the video output mcuh. Finally adjusting the threshold and the number of frames to aggregate over
did show improvement, but there is a lot of freedom these parameters, and the trial and error process I took will likely not generalize to a different video. Altogether, the whole process
is mostly trial-and-error and I believe new ideas/data (e.g. using lidars) are needed to make the vehicle detection work in the limitless variety of situations in the real world. 
For one, I like to include additional training images to improve the classification power of the model for the future.

The pipeline I built has a few limitations:
* We only search the image pixels with y-coordinates between 400 and 656. So it would fail if the camera isn't positioned in a way that the road ahead lies between
this range. This may also be a problem 
* The algorithm may fail if the frames are impacted by poor lighting conditions (say if it's at night) or if there is a lot of noise (say if it's raining heavily).
* Also, the pipeline is not good at all at detecting vehicles that are not in the training data, say motorcycles, 18 wheelers, bicycles, etc. Again increasing the
training data set will help, but at a certain point it may be worth using a convolutional neural network for the model instead of SVM or decision trees and image
processing techniques.

To make the pipeline more robust, we can train the vehicle classification model to be able to classify vehicles in all conditions. In the end, the better we can 
correctly classify images of vehicles as such, the better the whole pipeline will work. Of course, adding addtional sensor data beyond the front camera video will 
help a lot. 
