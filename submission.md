**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # "Image References"
[image1]: /output_images/car.png	""Car""
[image2]: /output_images/not_car.png	""Not Car""
[image3]: /output_images/hog.png
[image4]: /output_images/scale.png	""Different scales (2.5,2,1.5,1)""
[image5]: /output_images/nobox.png
[image6]: /output_images/index.png
[image7]: /output_images/heat_final.png
[image8]: /output_images/video.png
[image9]: /output_images/boxes.png
[link1]: https://www.youtube.com/watch?v=l0lzQY3TmCs&amp;amp;feature=youtu.be

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README
You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook `vehicle_detect.ipynb`  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the HOG parameters of `orientations=(11,9)`, `pixels_per_cell=(12, 8)` and `cells_per_block=(2, 2)`:


![alt text][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters for HOG and finally came up with the best choice based on the training accuracy obtained. The final HOG parameters were

`color_space = 'YCrCb' `
`orient = 9  `
`pix_per_cell = 12 `
`cell_per_block = 2 `
`hog_channel = "ALL" `
`spatial_size = (32, 32) `
`hist_bins = 32  `

`spatial_feat = True`

 `hist_feat = True `
`hog_feat = True `

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

For training the classifier, I used `GridSearchCV` with `SVM` classifier. The parameters for `SVM` were `'kernel':('linear', 'rbf'), 'C':[1, 10]`. The `kernel:rbf` and `C:10` were the best parameters. For obtaining features, I used`extract_features` function to features of `cars` and `notcars`. While extracting features, I used only `histogram` and `hog features`. I neglected the `spatial features` as the processing time escalated drastically, without much improvement in performance. The Test Accuracy of SVC was 0.995(or 99.5%), which is a good performance index. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Initially I tried different `spatial_size`and (32,32) worked best for me.  To remove unnecessary search area  in the image, I used `y_start	` and `y_stop`.At first, I tried only 1 scale, which gave satisfactory results only for some of the `test_images`. Therefore I came up with the idea of combining different image scales, each with different `y_start	` and `y_stop`. The different scales were `(1.5,1)`

![alt text][image4]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using RBG 3-channel HOG features and histograms of color in the feature vector, which provided a nice result. I realized that the scale of `2.0` was unnecessary as the detections were repetitive and time consuming. Therefore I decided to use the scales `1.5 and 1.0`which improved the performance.![alt text][image5]

![alt text][image9]

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video  result][link1]


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap for the test image.

![alt text][image6]

To deal with the False positives, I calculated the area of the blobs and put a threshold of min Area of blob   `> 3000 sqpx`.  I am maintaining the previous using a `deque` named `boxes`. Also I took cumulative heat map, by adding labels from past `5 frames` and keeping an varying threshold of `3.2*len(boxes)`. I came with the threshold by trail and error, and also for `1 frame` an individual of threshold of `3` resulted in desired output, as shown below

![alt text][image7]

Here is the final image from the Video, that takes average of `5` frames with threshold of `(5*3.2)`, which is quite similar to the above one.

![alt text][image8]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

​	This is my last project in the module 1 of Self Driving Nano degree. I thank Udacity for its marvelous content and mentorship. This has by far been the most challenging project for me. Tuning the parameters to get accuracy was initial challenge. Initially I used only HOG features with single scale, which led to lot of False positives even with high accuracy. The pipeline is designed only to identity cars. Therefore trucks and two wheelers will not be detected. In order to detect those as well, we can augment the dataset and retrain our model. Use of multiple scales with varied `y_start	` and `y_stop` improved the model performance drastically. Although this approach is working decently good, it might fail given a completely new scenario. Therefore, Deep learning techniques might be of more advantage. 

​	I observed that the algorithm is able to detect vehicles coming in the opposite direction, but due averaging over 5 frames, the detections are not in place they are expected as the relative velocity is quite for these. Therefore averaging with some correction is needed to give more weightage to recent frames.

​	