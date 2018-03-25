import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import pickle
import numpy as np
import glob

dist_pickle = pickle.load(open('calibration.p', 'rb'))
mtx = dist_pickle['mtx']
dist = dist_pickle['dist']

image = glob.glob('./test_images/test*.jpg')

for img in image:
	image = cv2.imread(img)
	out = cv2.undistort(image, mtx, dist, None, mtx)
	#print(img)
	name = './output_images/undist_'+img[14:]
	name2 = './output_images/plot_'+img[14:]
	#print(name)
	cv2.imwrite(name, out)
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
	f.tight_layout()
	ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	ax1.set_title('Origional Image')
	ax2.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
	ax2.set_title('Undistorted Image')
	f.savefig(name2)