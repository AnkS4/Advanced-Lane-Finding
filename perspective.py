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
dst = np.float32([[300, img_max[1]], [980, img_max[1]], [980, 0], [300, 0]])

#Reading each image and selecting source points manually 
path = './output_images/combined_straight_lines1.png'
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
src = np.float32([[210, img_max[1]], [1080, img_max[1]], [700, 455], [585, 455]])
warped = warp_img(src, dst, img)
name = './output_images/perspective' + path[24:]
mpimg.imsave(name, warped, cmap='gray')

path = './output_images/combined_straight_lines2.png'
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
src = np.float32([[225, img_max[1]], [1085, img_max[1]], [730, 475], [555, 475]])
warped = warp_img(src, dst, img)
name = './output_images/perspective' + path[24:]
mpimg.imsave(name, warped, cmap='gray')

path = './output_images/combined_test1.png'
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
src = np.float32([[265, img_max[1]], [1125, img_max[1]], [725, 460], [595, 460]])
warped = warp_img(src, dst, img)
name = './output_images/perspective' + path[24:]
mpimg.imsave(name, warped, cmap='gray')

path = './output_images/combined_test2.png'
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
src = np.float32([[300, img_max[1]], [1175, img_max[1]], [745, 480], [560, 480]])
warped = warp_img(src, dst, img)
name = './output_images/perspective' + path[24:]
mpimg.imsave(name, warped, cmap='gray')

path = './output_images/combined_test3.png'
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
src = np.float32([[245, img_max[1]], [1120, img_max[1]], [760, 480], [580, 480]])
warped = warp_img(src, dst, img)
name = './output_images/perspective' + path[24:]
mpimg.imsave(name, warped, cmap='gray')

path = './output_images/combined_test4.png'
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
src = np.float32([[280, img_max[1]], [1160, img_max[1]], [735, 465], [590, 465]])
warped = warp_img(src, dst, img)
name = './output_images/perspective' + path[24:]
mpimg.imsave(name, warped, cmap='gray')

path = './output_images/combined_test5.png'
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
src = np.float32([[180, img_max[1]], [1105, img_max[1]], [840, 530], [470, 530]])
warped = warp_img(src, dst, img)
name = './output_images/perspective' + path[24:]
mpimg.imsave(name, warped, cmap='gray')

path = './output_images/combined_test6.png'
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
src = np.float32([[270, img_max[1]], [1125, img_max[1]], [745, 465], [605, 465]])
warped = warp_img(src, dst, img)
name = './output_images/perspective' + path[24:]
mpimg.imsave(name, warped, cmap='gray')