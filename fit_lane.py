import cv2
import numpy as np 
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
import glob
import pickle

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

ymax = 719

images = glob.glob('./output_images/perspective*.png')
lane_pickle = {}

for i in images:
	img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)

	histogram = np.sum(img[img.shape[0]//2:, :], axis=0)

	mid = np.int(histogram.shape[0]//2)
	leftx_base = np.argmax(histogram[:mid])
	rightx_base = np.argmax(histogram[mid:]) + mid

	nwindows = 9
	window_height = np.int(img.shape[0]//nwindows)

	nonzero = img.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])

	leftx_current = leftx_base
	rightx_current = rightx_base

	margin = 100
	minpix = 50

	left_lane_inds = []
	right_lane_inds = []

	for window in range(nwindows):
	    win_y_low = img.shape[0] - (window+1)*window_height
	    win_y_high = img.shape[0] - window*window_height
	    win_xleft_low = leftx_current - margin
	    win_xleft_high = leftx_current + margin
	    win_xright_low = rightx_current - margin
	    win_xright_high = rightx_current + margin
	    # Draw the windows on the visualization image
	    cv2.rectangle(img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high), (0,255,0), 2) 
	    cv2.rectangle(img,(win_xright_low,win_y_low),(win_xright_high,win_y_high), (0,255,0), 2) 
	    # Identify the nonzero pixels in x and y within the window
	    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
	    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
	    # Append these indices to the lists
	    left_lane_inds.append(good_left_inds)
	    right_lane_inds.append(good_right_inds)
	    # If you found > minpix pixels, recenter next window on their mean position
	    if len(good_left_inds) > minpix:
	        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
	    if len(good_right_inds) > minpix:        
	        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
	right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

	# Generate x and y values for plotting
	ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	#Dump pickle for left_fitx, right_fitx
	lane_pickle['l' + i[28:-4]] = left_fitx
	lane_pickle['r' + i[28:-4]] = right_fitx

	left_center = left_fit[0]*ymax**2 + left_fit[1]*ymax + left_fit[2]
	right_center = right_fit[0]*ymax**2 + right_fit[1]*ymax + right_fit[2]
	lane_center = (left_center + right_center)/2
	camera_center = img.shape[1]/2
	lane_distance = (camera_center - lane_center) * ym_per_pix

	#Dump pickle for lane_distance
	lane_pickle['dist' + i[28:-4]] = lane_distance

	# Create an image to draw on and an image to show the selection window
	out_img = np.dstack((img, img, img))*255
	window_img = np.zeros_like(out_img)

	# Color in left and right line pixels
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

	# Generate a polygon to illustrate the search window area
	# And recast the x and y points into usable format for cv2.fillPoly()
	left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
	left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
	left_line_pts = np.hstack((left_line_window1, left_line_window2))
	right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
	right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
	right_line_pts = np.hstack((right_line_window1, right_line_window2))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
	cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
	result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
	
	name = './output_images/fit_lane' + i[27:]
	mpimg.imsave(name, result)
	name = './output_images/lane' + i[27:]
	mpimg.imsave(name, window_img)

	leftx = left_fitx[::-1]  # Reverse to match top-to-bottom in y
	rightx = right_fitx[::-1]  # Reverse to match top-to-bottom in y

	# Fit a second order polynomial to pixel positions in each lane line
	left_fit = np.polyfit(ploty, leftx, 2)
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fit = np.polyfit(ploty, rightx, 2)
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Define y-value where we want radius of curvature
	# Choose the maximum y-value, corresponding to the bottom of the image
	y_eval = np.max(ploty)
	#Radius of curvature in pixels
	left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
	right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

	# Fit new polynomials to x,y in world space
	left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
	right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
	# Calculate the new radii of curvature
	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

	# Radius of curvature in meters
	print('Curvature for', i[16:], 'is :', left_curverad, 'm', right_curverad, 'm')
	print('Distance of the vehicle from center is :', lane_distance, 'm')
	lane_pickle['lc' + i[28:-4]] = left_curverad
	lane_pickle['rc' + i[28:-4]] = right_curverad
	pickle.dump(lane_pickle, open('./lane.p', 'wb'))