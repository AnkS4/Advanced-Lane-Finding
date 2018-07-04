import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import glob

def warp_img(src, dst, img):
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

#Manual calculation of image Size and fixed destination (same for all images)
img_max = 1279, 719
img_size = 1280, 720
dst = np.float32([[200, img_max[1]], [1080, img_max[1]], [1080, 0], [200, 0]])

img = mpimg.imread('./output_images/combined8.png')
src = np.float32([[225, img_max[1]], [1135, img_max[1]], [725, 450], [625, 450]])
warped = warp_img(src, dst, img)
mpimg.imsave('./output_images/perspective8.png', warped)

img = mpimg.imread('./output_images/combined7.png')
src = np.float32([[260, img_max[1]], [1150, img_max[1]], [715, 450], [615, 450]])
warped = warp_img(src, dst, img)
mpimg.imsave('./output_images/perspective7.png', warped)

img = mpimg.imread('./output_images/combined6.png')
src = np.float32([[265, img_max[1]], [1130, img_max[1]], [715, 440], [620, 440]])
warped = warp_img(src, dst, img)
mpimg.imsave('./output_images/perspective6.png', warped)

img = mpimg.imread('./output_images/combined5.png')
src = np.float32([[265, img_max[1]], [1160, img_max[1]], [740, 465], [605, 465]])
warped = warp_img(src, dst, img)
mpimg.imsave('./output_images/perspective5.png', warped)

img = mpimg.imread('./output_images/combined4.png')
src = np.float32([[175, img_max[1]], [1120, img_max[1]], [735, 460], [600, 460]])
warped = warp_img(src, dst, img)
mpimg.imsave('./output_images/perspective4.png', warped)

img = mpimg.imread('./output_images/combined3.png')
src = np.float32([[230, img_max[1]], [1095, img_max[1]], [710, 465], [580, 465]])
warped = warp_img(src, dst, img)
mpimg.imsave('./output_images/perspective3.png', warped)

img = mpimg.imread('./output_images/combined2.png')
src = np.float32([[275, img_max[1]], [1120, img_max[1]], [725, 460], [590, 460]])
warped = warp_img(src, dst, img)
mpimg.imsave('./output_images/perspective2.png', warped)

img = mpimg.imread('./output_images/combined1.png')
src = np.float32([[300, img_max[1]], [1140, img_max[1]], [680, 450], [570, 450]])
warped = warp_img(src, dst, img)
mpimg.imsave('./output_images/perspective1.png', warped)