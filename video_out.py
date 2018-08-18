import cv2
import numpy as np 
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
import pickle
from moviepy.editor import VideoFileClip

i = 0
lane_found = False
left_fit_old = []
right_fit_old = []

def abs_sobel_thresh(abs_sobelx, abs_sobely, orient='x', thresh=(0, 255)):
    if orient == 'x':
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    if orient == 'y':
        scaled_sobel = np.uint8(255*abs_sobely/np.max(abs_sobely))

    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary

# Outputs magnitude of the gradient in the given threshold
def mag_thresh(sobelx, sobely, thresh=(0, 255)):
    abs_sobelxy = np.sqrt(sobelx**2 + sobely**2)

    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))

    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return mag_binary

# Outputs direction of the gradient in the given threshold
def dir_threshold(abs_sobelx, abs_sobely, thresh=(0, np.pi/2)):
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)

    dir_binary = np.zeros_like(grad_dir)
    dir_binary[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    return dir_binary

def threshold(ch, thresh=(0, 255)):
    binary = np.zeros_like(ch)
    binary[(ch > thresh[0]) & (ch <= thresh[1])] = 1
    return binary

def window_lane(warped, nonzeroy, nonzerox):
	#print('Sliding window')
	global lane_found, left_fit_old, right_fit_old

	margin = 100
	histogram = np.sum(warped[warped.shape[0]//2:, :], axis=0)

	mid = np.int(histogram.shape[0]//2)
	leftx_base = np.argmax(histogram[:mid])
	rightx_base = np.argmax(histogram[mid+1:]) + mid

	nwindows = 9
	window_height = np.int(warped.shape[0]//nwindows)

	leftx_current = leftx_base
	rightx_current = rightx_base
	left_lane_inds = []
	right_lane_inds = []
	minpix = 50

	for window in range(nwindows):
	    win_y_low = warped.shape[0] - (window+1)*window_height
	    win_y_high = warped.shape[0] - window*window_height
	    win_xleft_low = leftx_current - margin
	    win_xleft_high = leftx_current + margin
	    win_xright_low = rightx_current - margin
	    win_xright_high = rightx_current + margin
	    # Draw the windows on the visualization image
	    cv2.rectangle(warped,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high), (0,255,0), 2) 
	    cv2.rectangle(warped,(win_xright_low,win_y_low),(win_xright_high,win_y_high), (0,255,0), 2) 
	    # Identify the nonzero pixels in x and y within the window
	    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & 
	    	(nonzerox < win_xleft_high)).nonzero()[0]
	    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & 
	    	(nonzerox < win_xright_high)).nonzero()[0]
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

	left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & 
		(nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
	right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & 
		(nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

	# Generate x and y values for plotting
	ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	left_fit_old = left_fit
	right_fit_old = right_fit

	return left_fit, left_fitx, right_fit, right_fitx

def prev_lane(warped, nonzeroy, nonzerox):
	#print('Previous Lane')
	global lane_found, left_fit_old, right_fit_old
	margin = 100

	left_lane_inds = ((nonzerox > (left_fit_old[0]*(nonzeroy**2) + left_fit_old[1]*nonzeroy + left_fit_old[2] - margin))
		& (nonzerox < (left_fit_old[0]*(nonzeroy**2) + left_fit_old[1]*nonzeroy + left_fit_old[2] + margin)))
	right_lane_inds = ((nonzerox > (right_fit_old[0]*(nonzeroy**2) + right_fit_old[1]*nonzeroy + right_fit_old[2] - margin))
		& (nonzerox < (right_fit_old[0]*(nonzeroy**2) + right_fit_old[1]*nonzeroy + right_fit_old[2] + margin)))

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	r_range = (np.min(rightx) >= 640) & (np.max(rightx) <= 1280)
	l_range = (np.min(leftx) >= 0) & (np.max(leftx) <= 640)
	in_range = r_range & l_range

	if(in_range):
		# Fit a second order polynomial to each
		left_fit = np.polyfit(lefty, leftx, 2)
		right_fit = np.polyfit(righty, rightx, 2)

		# Generate x and y values for plotting
		ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
		left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

		left_fit_old = left_fit
		right_fit_old = right_fit
	else:
		left_fit, left_fitx, right_fit, right_fitx = window_lane(warped, nonzeroy, nonzerox)

	return left_fit, left_fitx, right_fit, right_fitx

def process_img(img):
	dist_pickle = pickle.load(open('calibration.p', 'rb'))
	mtx = dist_pickle['mtx']
	dist = dist_pickle['dist']

    # Define conversions in x and y from pixels space to meters
	ym_per_pix = 30/720 #320 # meters per pixel in y dimension
	xm_per_pix = 3.7/700 # meters per pixel in x dimension
	
	out = cv2.undistort(img, mtx, dist, None, mtx)
	
	r = out[:, :, 0]
	hls = cv2.cvtColor(out, cv2.COLOR_RGB2HLS)
	h = hls[:, :, 0]
	s = hls[:, :, 2]
	lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
	b = lab[:, :, 2]
	luv = cv2.cvtColor(out, cv2.COLOR_RGB2LUV)
	l = luv[:, :, 0]

	k = 15
	sobelx = cv2.Sobel(s, cv2.CV_64F, 1, 0, ksize=k) # Derivative in X direction
	sobely = cv2.Sobel(s, cv2.CV_64F, 0, 1, ksize=k) # Derivative in Y direction
	abs_sobelx = np.absolute(sobelx) #Absolute value of X derivative
	abs_sobely = np.absolute(sobely) #Absolute value of Y derivative

	r_binary = threshold(r, thresh=(175, 255))
	b_binary = threshold(b, thresh=(155, 225))
	h_binary = threshold(h, thresh=(10, 100))
	l_binary = threshold(l, thresh=(220, 255))
	s_binary = threshold(s, thresh=(180, 255))

	gradx = abs_sobel_thresh(abs_sobelx, abs_sobely, orient='x', thresh=(20, 100))
	grady = abs_sobel_thresh(abs_sobelx, abs_sobely,  orient='y', thresh=(40, 90))
	mag_binary = mag_thresh(sobelx, sobely, thresh=(30, 100))
	dir_binary = dir_threshold(abs_sobelx, abs_sobely, thresh=(0.7, 1.2))

	combined = np.zeros_like(mag_binary)
	combined[((gradx == 1) & (grady == 1) & (r_binary==1)) | ((h_binary==1) & (s_binary==1)) | (b_binary==1) | (l_binary==1)] = 1

	img_size = out.shape[1], out.shape[0]
	ymax = img_size[0]-1

	offset = 120
	offset2 = 200
	midh = img_size[0]/2
	topv = img_size[1]*(2/3)
	src = np.float32([[200, img_size[1]], [1150, img_size[1]], [760, topv], [560, topv]])
	dst = np.float32([[offset2, img_size[1]], [img_size[0]-offset2, img_size[1]], [img_size[0]-offset2, 0], [offset2, 0]])
	
	#Perform perspective transform
	M = cv2.getPerspectiveTransform(src, dst)
	warped = cv2.warpPerspective(combined, M, img_size, flags=cv2.INTER_LINEAR)

	nonzero = warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])

	global lane_found
	if not lane_found:
		left_fit, left_fitx, right_fit, right_fitx = window_lane(warped, nonzeroy, nonzerox)
		lane_found = True
	else:
		left_fit, left_fitx, right_fit, right_fitx = prev_lane(warped, nonzeroy, nonzerox)

	left_center = left_fit[0]*ymax**2 + left_fit[1]*ymax + left_fit[2]
	right_center = right_fit[0]*ymax**2 + right_fit[1]*ymax + right_fit[2]
	lane_center = np.mean((left_fitx + right_fitx)/2)
	camera_center = img_size[0]/2
	lane_distance = (camera_center - lane_center) * xm_per_pix

	# Create an image to draw on and an image to show the selection window
	out_img = np.dstack((warped, warped, warped))*255

	leftx = left_fitx[::-1]  # Reverse to match top-to-bottom in y
	rightx = right_fitx[::-1]  # Reverse to match top-to-bottom in y

	ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
	# Fit a second order polynomial to pixel positions in each lane line
	left_fit = np.polyfit(ploty, leftx, 2)
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fit = np.polyfit(ploty, rightx, 2)
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Fit new polynomials to x,y in world space
	left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
	right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

	y_eval = np.max(ploty)
	# Calculate the new radii of curvature
	l_roc = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	r_roc = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

	#Start warping lane on undistorted image
	roc = round((l_roc + r_roc)/2, 2)
	dist = round(lane_distance, 2)
	abs_dist = abs(dist)

	warp_zero = np.zeros_like(out).astype(np.uint8)

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(warp_zero, np.int_([pts]), (0, 255, 0))

	Minv = cv2.getPerspectiveTransform(dst, src)
	newwarp = cv2.warpPerspective(warp_zero, Minv, (img.shape[1], img.shape[0])) 
	result = cv2.addWeighted(out, 1, newwarp, 0.3, 0)

	#Put the text on the image
	text = 'Radius of curvature = ' + str(roc) + 'm'
	cv2.putText(result, text, org=(0, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
	if dist > 0:
		text2 = 'Vehicle is '+ str(abs_dist) + 'm right from the center'
	elif dist < 0:
		text2 = 'Vehicle is '+ str(abs_dist) + 'm left from the center'
	else:
		text2 = 'Vehicle is exactly at the center'
	cv2.putText(result, text2, org=(0, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

	return result

vid = VideoFileClip('project_video.mp4')
out_vid = vid.fl_image(process_img)
out_vid.write_videofile('output_video.mp4', audio=False)