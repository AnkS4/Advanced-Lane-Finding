import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import glob
from matplotlib.gridspec import GridSpec

images = glob.glob('./output_images/undist_*.jpg')

# Outputs absolute threshold value respective to the passed orientation
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

def hls_threshold(arr, thresh=(0, 255)):
    binary = np.zeros_like(arr)
    binary[(arr > thresh[0]) & (arr <= thresh[1])] = 1
    return binary

def s_threshold(s, thresh=(150, 255)):
    s_binary = np.zeros_like(s)
    s_binary[(s > thresh[0]) & (s <= thresh[1])] = 1
    return s_binary

for img in images:
    image = mpimg.imread(img)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s = hls[:, :, 2]

    k = 15
    sobelx = cv2.Sobel(s, cv2.CV_64F, 1, 0, ksize=k) # Derivative in X direction
    sobely = cv2.Sobel(s, cv2.CV_64F, 0, 1, ksize=k) # Derivative in Y direction
    abs_sobelx = np.absolute(sobelx) #Absolute value of X derivative
    abs_sobely = np.absolute(sobely) #Absolute value of Y derivative

    s_binary = s_threshold(s, thresh=(150, 255))
    gradx = abs_sobel_thresh(abs_sobelx, abs_sobely, orient='x', thresh=(50, 255))
    grady = abs_sobel_thresh(abs_sobelx, abs_sobely,  orient='y', thresh=(50, 255))
    mag_binary = mag_thresh(sobelx, sobely, thresh=(75, 255))
    dir_binary = dir_threshold(abs_sobelx, abs_sobely, thresh=(0.6, 1.3))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (s_binary == 1)] = 1

    f = plt.figure()
    gs = GridSpec(12, 12)

    ax1 = f.add_subplot(gs[0:4, 0:4])
    ax2 = f.add_subplot(gs[4:8, 0:4])
    ax3 = f.add_subplot(gs[8:12, 0:4])
    ax4 = f.add_subplot(gs[0:6, 4:8])
    ax5 = f.add_subplot(gs[6:12, 4:8])
    ax6 = f.add_subplot(gs[0:6, 8:12])
    ax7 = f.add_subplot(gs[6:12, 8:12])

    ax1.imshow(gradx, cmap='gray')
    ax1.set_title('Absolute Gradient X')
    ax2.imshow(grady, cmap='gray')
    ax2.set_title('Absolute Gradient Y')
    ax3.imshow(mag_binary, cmap='gray')
    ax3.set_title('Magnitude of Gradient')
    ax4.imshow(dir_binary, cmap='gray')
    ax4.set_title('Direction of Gradient')
    ax5.imshow(s_binary, cmap='gray')
    ax5.set_title('S Channel')
    ax6.imshow(image)
    ax6.set_title('Original Image')
    ax7.imshow(combined, cmap='gray')
    ax7.set_title('Combined Threshold')
    name = './output_images/thresh_plot_'+img[23:-3]+'png'
    gs.tight_layout(f)
    f.savefig(name)

    name = './output_images/magnitude_'+img[23:-3]+'png'
    mpimg.imsave(name, mag_binary, cmap='gray')
    name = './output_images/combined_'+img[23:-3]+'png'
    mpimg.imsave(name, combined, cmap='gray')