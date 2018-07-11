[//]: # (Image References)

[chess]: ./camera_cal/calibration3.jpg "Chessboard"
[corners]: ./output_images/calibration3.jpg "Corners"
[dist]: ./test_images/test1.jpg "Distorted Image"
[undist]: ./output_images/undist_test1.jpg "Undistorted Image"
[thresh]: ./output_images/thresh_plot_straight_lines1.png
[img]: ./output_images/combined_test1.png
[imgp]: ./output_images/perspective_test1.png
[fit]: ./output_images/fit_lane_test1.png

# Advanced-Lane-Finding
Finding lane lines on the road.

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## I did the project as follows:

### 1. Camera Calibration:

I used *cv2.findChessboardCorners()* for finding all the corners in the given image. I saved these corners from all the read images to imgpoints. Using *cv2.calibrateCamera()* with obtained imgpoints and real world co-ordinates, it outputs **Camera Matrix**, **Distortion Coefficient**, **Rotational Vectors**, **Translation vectors** with which we can undistort any image captured from the same camera.

One of the image with detected corners:

| Chessboard Image          | Detected Corners               |
|:-------------------------:|:------------------------------:|
| ![Chessboard][chess]      | ![Chessboard Corners][corners] |

### 2. Distortion Correction

I used **Camera Matrix** and **Distortion Coefficient** from previous output to undistort the image using *cv2.undistort()*.
The images may seem to be similar after undistortion. 

The difference can be seem at the edges the image for the following example.

Image after undistortion:

|  Original Image             |  Undistorted Image             |
|:---------------------------:|:------------------------------:|
| ![Original Image][dist]     | ![Undistorted Image][undist]   |


### 3. Thresholding

I tried various combinations from *S Channel* from HLS color space, *R Channel*, *G Channel*, *B Channel* from RGB color space, *Sobel Gradient in X*,  *Sobel Gradient in Y*, *Sobel Magnitude* & *Sobel Direction*.

Among them *S Channel* gave me the most accurate detection. I used **combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (s_binary == 1)] = 1** combination for increasing the accuracy of the detection.

One of the threshold plot:

|  Threshold Plot             |  
|:---------------------------:|
| ![Threshold Plot][thresh]   |

### 4. Perspective Transform

For perspective/birds-eye view transform, I manually selected the co-ordinates for each image from thresholding output.

Perspective output from one of the image:

|  Original Image             |  Undistorted Image             |  Thresholded Image          |  Perspective Image             |
|:---------------------------:|:------------------------------:|:---------------------------:|:------------------------------:|
| ![Original Image][dist]     | ![Undistorted Image][undist]   | ![Thresholded Image][img]   | ![Perspective Image][imgp]     |


### 5. Polynomial Fitting

I used the output from perspective for fitting the lane. Using histogram of the perspective image, I detected the left & right base using *np.argmax()*. Starting from that initial points left and right points are collected by deciding specific number of windows. Each window has left and right points correspoding to a lane. Then, polynomial corresponding to each lane is obtained by *np.polyfit()*, it's is used for fitting the lane.

Example of lane fitting:

|  Original Image             |  Perspective Image          |  Lane Fit                      |
|:---------------------------:|:---------------------------:|:------------------------------:|
| ![Original Image][dist]     | ![Perspective Image][imgp]  | ![Lane Fit ][fit]              |


### 6. Calculating Radius of Curvature

The radius of curvature in pixels is calculated using *left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])^2)^1.5) / np.absolute(2*left_fit[0])*, *right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])^2)^1.5) / np.absolute(2*right_fit[0])* 

Later, it is converted to meters.

Output for the images is below:

| Image Name                  | Left Lane Curvature         | Right Lane Curvature           |
|:---------------------------:|:---------------------------:|:------------------------------:|
| perspective_test6           | 3234.5940733240163 m        | 983.7502369781216 m            |
| perspective_test5           | 27420.065667106228 m        | 2625.5409849369294 m           |
| perspective_test4           | 4448.201716046439 m         | 1209.9983366753997 m           |
| perspective_test1           | 920.1252941991785 m         | 2322.9544873268974 m           |
| perspective_straight_lines2 | 1575.8250305220645 m        | 14180.48921341102 m            |
| perspective_straight_lines1 | 2577.6586892197365 m        | 225.41822384915173 m           |
| perspective_test3           | 3378.061657403786 m         | 1094.8882103375329 m           |
| perspective_test2           | 2088.473302866544 m         | 372.87185571782027 m           |
