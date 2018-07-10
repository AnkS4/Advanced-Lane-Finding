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

I used cv2.findChessboardCorners() for finding all the corners in the given image. I saved these corners from all the read images to imgpoints & using cv2.calibrateCamera() on them gives 'Camera Matrix', 'Distortion Coefficient', 'Rotational Vectors', 'Translation vectors'.

### 2. Distortion Correction

### 3. Thresholding

### 4. Perspective Transform

### 5. Polynomial Fitting

### 6. Calculating Radius of Curvature
