import cv2
import numpy as np 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
import glob
import pickle


images = glob.glob('./camera_cal/calibration*.jpg')

objpoints = [] # 3D Real world coordinates
imgpoints = [] # 2D Image coordinatives

objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)


for image in images:
	img = mpimg.imread(image)
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	ret, corners = cv2.findChessboardCorners(img, (9, 6), None)

	if ret == True:
		imgpoints.append(corners)
		objpoints.append(objp)

		img2 = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
		name = './output_images/' + image[13:]
		cv2.imwrite(name, img2)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

dist_pickle = {}
dist_pickle['mtx'] = mtx
dist_pickle['dist'] = dist
pickle.dump(dist_pickle, open('./calibration.p', 'wb'))