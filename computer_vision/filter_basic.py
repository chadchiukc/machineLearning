from scipy import ndimage
from PIL import Image
import numpy as np


def read_img_as_array(file):
    '''read image and convert it to numpy array with dtype np.float64'''
    img = Image.open(file)
    arr = np.asarray(img, dtype=np.float64)
    return arr


def show_array_as_img(arr):
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:  # make sure arr falls in [0,255]
        arr = (arr - min)/(max-min)*255

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.show()


def save_array_as_img(arr, file):
    # make sure arr falls in [0,255]
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:
        arr = (arr - min) / (max - min) * 255

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(file)


def sharpened_image(file, sigma=3, alpha=1):
    arr = read_img_as_array(file)
    gaussian_filtered_arr = ndimage.gaussian_filter(arr, sigma=sigma)
    detailed_arr = arr - gaussian_filtered_arr
    sharpened_arr = arr + alpha * detailed_arr
    sharpened_arr[sharpened_arr < 0] = 0
    sharpened_arr[sharpened_arr > 255] = 255
    return sharpened_arr


def median_filter(file, filter_size):
    if filter_size % 2 == 0:
        raise ValueError('the filter size cannot be a even number')
    file_arr = read_img_as_array(file)
    ind = filter_size // 2
    r_arr = file_arr[:, :, 0]
    g_arr = file_arr[:, :, 1]
    b_arr = file_arr[:, :, 2]
    x, y = file_arr.shape[0:2]
    rgb_output = [np.zeros((x, y)), np.zeros((x, y)), np.zeros((x, y))]
    for index, rgb_arr in enumerate([r_arr, g_arr, b_arr]):
        for i in range(x):
            for j in range(y):
                temp = []
                for k in range(filter_size):
                    if i + k - ind < 0 or i + k - ind > x - 1:
                        for _ in range(filter_size):
                            temp.append(0)  # zero padding
                    else:
                        if j + k - ind < 0 or j + ind > y - 1:
                            temp.append(0)  # zero padding
                        else:
                            for z in range(filter_size):
                                temp.append(rgb_arr[i + k - ind][j + z - ind])

                temp.sort()
                rgb_output[index][i][j] = temp[len(temp) // 2]  # select the median of the temp

    return np.dstack((rgb_output[0], rgb_output[1], rgb_output[2]))


sharpened_rain = sharpened_image('rain.jpeg', alpha=3)
show_array_as_img(sharpened_rain)
save_array_as_img(sharpened_rain, 'sharpened.jpg')

derain = median_filter('rain.jpeg', 5)
show_array_as_img(derain)
save_array_as_img(derain, 'derained.jpg')



