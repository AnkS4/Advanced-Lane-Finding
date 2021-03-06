[//]: # (Image References)

[chess]: ./camera_cal/calibration3.jpg "Chessboard"
[corners]: ./output_images/calibration3.jpg "Corners"
[dist]: ./test_images/test1.jpg "Distorted Image"
[undist]: ./output_images/undist_test1.jpg "Undistorted Image"
[thresh]: ./output_images/thresh_plot_straight_lines1.png
[img]: ./output_images/combined_test1.png
[imgp]: ./output_images/perspective_test1.png
[fit]: ./output_images/fit_lane_test1.png
[l1]: ./output_images/fit_lane_straight_lines1.png
[l2]: ./output_images/fit_lane_straight_lines2.png
[l3]: ./output_images/fit_lane_test1.png
[l4]: ./output_images/fit_lane_test2.png
[l5]: ./output_images/fit_lane_test3.png
[l6]: ./output_images/fit_lane_test4.png
[l7]: ./output_images/fit_lane_test5.png
[l8]: ./output_images/fit_lane_test6.png
[org]: ./output_images/undist_test4.jpg
[warp]: ./output_images/line_plot_test4.png
[test1]: ./test_images/straight_lines1.jpg
[test2]: ./test_images/straight_lines2.jpg
[test3]: ./test_images/test1.jpg
[test4]: ./test_images/test2.jpg
[test5]: ./test_images/test3.jpg
[test6]: ./test_images/test4.jpg
[test7]: ./test_images/test5.jpg
[test8]: ./test_images/test6.jpg
[out1]: ./output_images/line_plot2_straight_lines1.png
[out2]: ./output_images/line_plot2_straight_lines2.png
[out3]: ./output_images/line_plot2_test1.png
[out4]: ./output_images/line_plot2_test2.png
[out5]: ./output_images/line_plot2_test3.png
[out6]: ./output_images/line_plot2_test4.png
[out7]: ./output_images/line_plot2_test5.png
[out8]: ./output_images/line_plot2_test6.png

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
| ![Image 1][l1]              | 2577.6586892197365 m        | 225.41822384915173 m           |
| ![Image 2][l2]              | 1575.8250305220645 m        | 14180.48921341102 m            |
| ![Image 3][l3]              | 920.1252941991785 m         | 2322.9544873268974 m           |
| ![Image 4][l4]              | 2088.473302866544 m         | 372.87185571782027 m           |
| ![Image 5][l5]              | 3378.061657403786 m         | 1094.8882103375329 m           |
| ![Image 6][l6]              | 4448.201716046439 m         | 1209.9983366753997 m           |
| ![Image 7][l7]              | 27420.065667106228 m        | 2625.5409849369294 m           |
| ![Image 8][l8]              | 3234.5940733240163 m        | 983.7502369781216 m            |

### 7. Warping the lane on the original image

One of the image with warped lane on top of it:

|  Original Image             |  Image with warped lane        |
|:---------------------------:|:------------------------------:|
| ![Original Image][org]      | ![Warped Image][warp]          |

### 8. Estimating lane curvature and vehicle position

All of the test images with visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position:

| Original Image                 |  Final Image Output            |
|:------------------------------:|:------------------------------:|
| ![Original Image][test1]       | ![Output Image][out1]          |
| ![Original Image][test2]       | ![Output Image][out2]          |
| ![Original Image][test3]       | ![Output Image][out3]          |
| ![Original Image][test4]       | ![Output Image][out4]          |
| ![Original Image][test5]       | ![Output Image][out5]          |
| ![Original Image][test6]       | ![Output Image][out6]          |
| ![Original Image][test7]       | ![Output Image][out7]          |
| ![Original Image][test8]       | ![Output Image][out8]          |

### Discussion

#### Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

Selecting right number of color channels, reading/writing appropriate color channels was issue at the start. Such as reading image using *mpimg.imread()*, *cv2.imread()*, *mpimg.imsave(), *cv2.imwrite()*. Reading grayscale image using cv2 was an issue for me, then I used *cv2.imread(i, cv2.IMREAD_GRAYSCALE)* for reading grayscale image.

Selecting *color spaces*, *threshold limit* for image pipeline and *source points* for the video warping was experimental, I had to manually check many parameters, values for optimal output.

I also didn't know which lane curvature measurement to put on the image left or right. Some images had significant difference between the lane. So, I put the average of left and right lane curvature as measurement.

On video pipeline, I faced errors like *TypeError("expected non-empty vector for x")*. One of the frame was not having detected right lane in warped lane, so *np.polyfit()* couldn't fit the right lane.


