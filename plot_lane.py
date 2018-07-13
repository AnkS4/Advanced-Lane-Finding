import cv2
import numpy as np 
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
import pickle

img_max = 1279, 719

lane_pickle = pickle.load(open('lane.p', 'rb'))
dst = np.float32([[300, img_max[1]], [980, img_max[1]], [980, 0], [300, 0]])
ploty = np.linspace(0, 719, 720)

def plot_img(i, src):
	#Read lane-only image
	#i = './output_images/lane_straight_lines1.png'
	img = mpimg.imread(i)
	#Read undistorted image
	undist_path = './output_images/undist_' + i[21:-4] + '.jpg'
	undist = mpimg.imread(undist_path, 1)

	#Read pickle
	left_fitx = lane_pickle['l' + undist_path[23:-4]]
	right_fitx = lane_pickle['r' + undist_path[23:-4]]
	#src = np.float32([[210, img_max[1]], [1080, img_max[1]], [700, 455], [585, 455]])
	warp_zero = np.zeros_like(undist).astype(np.uint8)

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(warp_zero, np.int_([pts]), (0, 255, 0))

	Minv = cv2.getPerspectiveTransform(dst, src)
	newwarp = cv2.warpPerspective(warp_zero, Minv, (img.shape[1], img.shape[0])) 
	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
	
	#Save result
	name = './output_images/line_plot_' + i[21:]
	mpimg.imsave(name, result)

#Image 1
i = './output_images/lane_straight_lines1.png'
src = np.float32([[210, img_max[1]], [1080, img_max[1]], [700, 455], [585, 455]])
plot_img(i, src)

#Image 2
i = './output_images/lane_straight_lines2.png'
src = np.float32([[225, img_max[1]], [1085, img_max[1]], [730, 475], [555, 475]])
plot_img(i, src)

#Image 3
i = './output_images/lane_test1.png'
src = np.float32([[265, img_max[1]], [1125, img_max[1]], [725, 460], [595, 460]])
plot_img(i, src)

#Image 4
i = './output_images/lane_test2.png'
src = np.float32([[300, img_max[1]], [1175, img_max[1]], [745, 480], [560, 480]])
plot_img(i, src)

#Image 5
i = './output_images/lane_test3.png'
src = np.float32([[245, img_max[1]], [1120, img_max[1]], [760, 480], [580, 480]])
plot_img(i, src)

#Image 6
i = './output_images/lane_test4.png'
src = np.float32([[280, img_max[1]], [1160, img_max[1]], [735, 465], [590, 465]])
plot_img(i, src)

#Image 7
i = './output_images/lane_test5.png'
src = np.float32([[180, img_max[1]], [1105, img_max[1]], [840, 530], [470, 530]])
plot_img(i, src)

#Image 8
i = './output_images/lane_test6.png'
src = np.float32([[270, img_max[1]], [1125, img_max[1]], [745, 465], [605, 465]])
plot_img(i, src)