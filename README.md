### **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./images/1.PNG
[image2]: ./images/2.PNG
[image3]: ./images/3.PNG
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

#### [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
##### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

This README wals you through the process I followed to complete this project. The code is in the form of a ipyhon notebook called Vehicle Detection And Tracking.ipynb in this repository.

### Data exploration and preparation

In the first 4 code cells of the notebook I am getting ready for the implementation by importing all the necessary python packages, helper functions and little bit of data exploration. I am using sklearn packages for the machine learning portion of the project. I am using Support Vector Machines for this. 

I use the provided Car and Non Car images as my test data. In code cell 3, I am loading the list of test data in to 2 lists called cars and notcars. It seems to be a pretty balanced data set with 8968 notcars and 8792 cars. 

In code cell 4, I am printing out few examples of each class of images. 

![alt text][image1]

#### Test Images

I use the proided test images ( in ./test_images folder) to tet out my pipeline. Below is an exampe of one of those images. 

![alt text][image2]

### Feature extraction

In the next few code cells, I use few methods to extract few different types of features for the training image to train the model on.
* Spacial Binning
* Color Histogram
* HOG features

#### Spacial Binning
Code cell 16 implements the function bin_spatial() that returns a collection of features created by performingthe spacial binning of the image. This function accepts the color space and size of the image tobe resized as parameters and return a flattened array of features. 

#### Color Histogram features

Code cell 17 implements the function color_hist() that returns color histogram features of the image. 

#### Histogram of Oriented Gradients (HOG)

Code cell 18 implments the function get_hog_features() to extract the HOG featurs of the image. This function accpepts parameters such as orient, pixels per cell and cells per block and utilizes skimage.feature.hog class to extract the HOG features. Below example shows the HOG representation of a car and a non car image.

![alt text][image3]

#### Putting all that together

In the code cell 20, I put all the feature together in the extract_features() function. Here, I get the list of images (cars or notcars) and a color space for Spacial binning and color histogram and the params for HOG mentioned above. The function will perform all the feature extraction methods explained in the previous secions and return a concatenated list of features in the form of a flattened array for training the model. Below are the list pf parameters I used in my final model after expriementing with many different combinations. (Code cell 22)

```python

color_space = 'YUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 16  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
y_start_stop = [400, 720]  # Min and max in y to search in slide_window()

```


I tried various combinations of parameters and before I settle down with te above set of params. I used the test accuracy score with different combinations of the parameters to decide on the final params. Below is the details of the model I trained.  

In Code cell 22 I created the featurs for Cars and notcars seperately and saved them in car_features and notcar_features lists. I then shuffle them before I use training test splis to create Training and test sets in code cell 24. The Training set contains 14208 and test se contains 3552.

I trained a linear SVM using the training set and I got the test accuracy of 98.96.

13.73 Seconds to train SVC...
Test Accuracy of SVC =  0.9896

#### Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

