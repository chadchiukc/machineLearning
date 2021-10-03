from PIL import Image # pillow package
import numpy as np
from scipy import ndimage

def read_img_as_array(file):
    '''read image and convert it to numpy array with dtype np.float64'''
    img = Image.open(file)
    arr = np.asarray(img, dtype=np.float64)
    return arr

def save_array_as_img(arr, file):
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:  # make sure arr falls in [0,255]
        arr = (arr - min)/(max-min)*255

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(file)

def show_array_as_img(arr, rescale='minmax'):
    
    # make sure arr falls in [0,255]
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:
        arr = (arr - min)/(max-min)*255

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.show()

def rgb2gray(arr):
    R = arr[:, :, 0] # red channel
    G = arr[:, :, 1] # green channel
    B = arr[:, :, 2] # blue channel
    gray = 0.2989*R + 0.5870*G + 0.1140*B
    return gray

#########################################
## Please complete following functions ##
#########################################


def median_filter(img, s):
    '''Perform median filter of size s x s to image 'arr', and return the filtered image.'''
    # TODO: Please complete this function.
    if s % 2 == 0:
        raise ValueError('the filter size cannot be a even number')
    ind = s // 2
    r_arr = img[:, :, 0]
    g_arr = img[:, :, 1]
    b_arr = img[:, :, 2]
    x, y = img.shape[0:2]
    rgb_output = [np.zeros((x, y)), np.zeros((x, y)), np.zeros((x, y))]
    for index, rgb_arr in enumerate([r_arr, g_arr, b_arr]):
        for i in range(x):
            for j in range(y):
                temp = []
                for k in range(s):
                    if i + k - ind < 0 or i + k - ind > x - 1:
                        for _ in range(s):
                            temp.append(0)  # zero padding
                    else:
                        if j + k - ind < 0 or j + ind > y - 1:
                            temp.append(0)  # zero padding
                        else:
                            for z in range(s):
                                temp.append(rgb_arr[i + k - ind][j + z - ind])

                temp.sort()
                rgb_output[index][i][j] = temp[len(temp) // 2]  # select the median of the temp
    arr = np.dstack((rgb_output[0], rgb_output[1], rgb_output[2]))
    return arr

def sharpen(img, sigma, alpha):
    '''Sharpen the image. 'sigma' is the standard deviation of Gaussian filter. 'alpha' controls how much details to add.'''
    # TODO: Please complete this function.
    gaussian_filtered_arr = ndimage.gaussian_filter(img, sigma=sigma)
    detailed_arr = img - gaussian_filtered_arr
    arr = img + alpha * detailed_arr
    arr[arr < 0] = 0
    arr[arr > 255] = 255
    return arr


img = read_img_as_array('rain.jpeg')
#TODO: finish assignment Part I.
sharpend_img = sharpen(img, 3, 10)
show_array_as_img(sharpend_img)
save_array_as_img(sharpend_img, 'sharpened.jpg')

median_filtered_img = median_filter(img, 5)
show_array_as_img(median_filtered_img)
save_array_as_img(median_filtered_img, 'derained.jpg')

