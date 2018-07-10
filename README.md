[//]: # (Image References)

[chess]: ./camera_cal/calibration3.jpg "Chessboard"
[corners]: ./output_images/calibration3.jpg "Corners"
[dist]: ./test_images/test1.jpg "Distorted Image"
[undist]: ./output_images/undist_test1.jpg "Undistorted Image"

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
The images may seem to be similar after undistortion. The difference can be seem at the edges the image.

Image after undistortion:

|  Original Image             |  Undistorted Image             |
|:---------------------------:|:------------------------------:|
| ![Original Image][dist]     | ![Undistorted Image][undist]   |


### 3. Thresholding

### 4. Perspective Transform

### 5. Polynomial Fitting

### 6. Calculating Radius of Curvature
