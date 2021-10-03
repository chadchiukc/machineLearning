from PIL import Image # pillow package
import numpy as np
from scipy import ndimage

def read_img_as_array(file):
    '''read image and convert it to numpy array with dtype np.float64'''
    img = Image.open(file)
    arr = np.asarray(img, dtype=np.float64)
    return arr


def save_array_as_img(arr, file):
    
    # make sure arr falls in [0,255]
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:
        arr = (arr - min)/(max-min)*255

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(file)


def show_array_as_img(arr):
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:  # make sure arr falls in [0,255]
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

def sobel(arr):
    '''Apply sobel operator on arr and return the result.'''
    # TODO: Please complete this function.
    # your code here
    vertical_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # vertical_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    horizontal_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Gx = ndimage.convolve(arr, vertical_filter)
    Gy = ndimage.convolve(arr, horizontal_filter)
    Gx *= 255 / Gx.max()
    Gy *= 255 / Gy.max()
    G = np.sqrt(np.square(Gx) + np.square(Gy))
    return G, Gx, Gy

def nonmax_suppress(G, Gx, Gy):
    '''Suppress non-max value along direction perpendicular to the edge.'''
    assert G.shape == Gx.shape
    assert G.shape == Gy.shape
    # TODO: Please complete this function.
    print(G.max())
    print(G.min())
    theta = np.arctan2(Gy, Gx) * 180 / np.pi
    theta[theta < 0] += 180
    x, y = G.shape
    suppressed_G = np.zeros((x, y), dtype=np.int32)

    for i in range(1, x - 1):
        for j in range(1, y - 1):
            if (0 <= theta[i, j] < 22.5) or (157.5 <= theta[i, j] <= 180):
                N1 = G[i, j + 1]
                N2 = G[i, j - 1]
            elif 22.5 <= theta[i, j] < 67.5:
                N1 = G[i + 1, j - 1]
                N2 = G[i - 1, j + 1]
            elif 67.5 <= theta[i, j] < 112.5:
                N1 = G[i + 1, j]
                N2 = G[i - 1, j]
            elif 112.5 <= theta[i, j] < 157.5:
                N1 = G[i - 1, j - 1]
                N2 = G[i + 1, j + 1]

            if (G[i, j] >= N1) and (G[i, j] >= N2):
                suppressed_G[i, j] = G[i, j]
            else:
                suppressed_G[i, j] = 0
    return suppressed_G

def thresholding(G, low, high):
    '''Binarize G according threshold low and high'''
    # TODO: Please complete this function.
    # your code here
    assert high <= 361
    assert low >= 0 & low < high
    x, y = G.shape
    result = np.zeros((x, y), dtype=np.int32)
    weak = 100
    strong = 360

    strong_i, strong_j = np.where(G >= high)
    weak_i, weak_j = np.where((G <= high) & (G >= low))

    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak

    # for i in range(1, x-1):
    #     for j in range(1, y-1):
    #         if result[i,j] == weak:
    #             if (result[i+1, j-1] == strong or result[i+1, j] == strong or result[i+1, j+1] == strong
    #                 or result[i, j-1] == strong or result[i, j+1] == strong
    #                 or result[i-1, j-1] == strong or result[i-1, j] == strong or result[i-1, j+1] == strong):
    #                 result[i, j] = strong
    #             else:
    #                 result[i, j] = 0
    # G = result

    top_to_bottom = result.copy()
    bottom_to_top = result.copy()
    right_to_left = result.copy()
    left_to_right = result.copy()
    image_row, image_col = result.shape
    for row in range(1, image_row):
        for col in range(1, image_col):
            if top_to_bottom[row, col] == weak:
                if top_to_bottom[row, col + 1] == strong or top_to_bottom[row, col - 1] == strong or top_to_bottom[
                    row - 1, col] == strong or top_to_bottom[
                    row + 1, col] == strong or top_to_bottom[
                    row - 1, col - 1] == strong or top_to_bottom[row + 1, col - 1] == strong or top_to_bottom[
                    row - 1, col + 1] == strong or top_to_bottom[
                    row + 1, col + 1] == strong:
                    top_to_bottom[row, col] = strong
                else:
                    top_to_bottom[row, col] = 0


    for row in range(image_row - 1, 0, -1):
        for col in range(image_col - 1, 0, -1):
            if bottom_to_top[row, col] == weak:
                if bottom_to_top[row, col + 1] == strong or bottom_to_top[row, col - 1] == strong or bottom_to_top[
                    row - 1, col] == strong or bottom_to_top[
                    row + 1, col] == strong or bottom_to_top[
                    row - 1, col - 1] == strong or bottom_to_top[row + 1, col - 1] == strong or bottom_to_top[
                    row - 1, col + 1] == strong or bottom_to_top[
                    row + 1, col + 1] == strong:
                    bottom_to_top[row, col] = strong
                else:
                    bottom_to_top[row, col] = 0


    for row in range(1, image_row):
        for col in range(image_col - 1, 0, -1):
            if right_to_left[row, col] == weak:
                if right_to_left[row, col + 1] == strong or right_to_left[row, col - 1] == strong or right_to_left[
                    row - 1, col] == strong or right_to_left[
                    row + 1, col] == strong or right_to_left[
                    row - 1, col - 1] == strong or right_to_left[row + 1, col - 1] == strong or right_to_left[
                    row - 1, col + 1] == strong or right_to_left[
                    row + 1, col + 1] == strong:
                    right_to_left[row, col] = strong
                else:
                    right_to_left[row, col] = 0

    for row in range(image_row - 1, 0, -1):
        for col in range(1, image_col):
            if left_to_right[row, col] == weak:
                if left_to_right[row, col + 1] == strong or left_to_right[row, col - 1] == strong or left_to_right[
                    row - 1, col] == strong or left_to_right[
                    row + 1, col] == strong or left_to_right[
                    row - 1, col - 1] == strong or left_to_right[row + 1, col - 1] == strong or left_to_right[
                    row - 1, col + 1] == strong or left_to_right[
                    row + 1, col + 1] == strong:
                    left_to_right[row, col] = strong
                else:
                    left_to_right[row, col] = 0
    final_image = top_to_bottom + bottom_to_top + right_to_left + left_to_right

    final_image[final_image > 255] = 255
    return final_image

def hough(G):
    '''Return Hough transform of G'''
    # TODO: Please complete this function.
    # your code here
    pass

img = read_img_as_array('road.jpeg')
#TODO: detect edges on 'img'
# question 1
gray_img = rgb2gray(img)
# show_array_as_img(gray_img)
save_array_as_img(gray_img, 'gray.jpg')

# question 2
gauss_img = ndimage.gaussian_filter(gray_img, sigma=1.3)
# show_array_as_img(gauss_img)
save_array_as_img(gauss_img, 'gauss.jpg')

# question 3
G, Gx, Gy = sobel(gauss_img)
show_array_as_img(G)
show_array_as_img(Gx)
# show_array_as_img(Gy)
save_array_as_img(G, 'G.jpg')
save_array_as_img(Gx, 'G_x.jpg')
save_array_as_img(Gy, 'G_y.jpg')

# question 4
nonmax_suppress_img = nonmax_suppress(G,Gx,Gy)
# show_array_as_img(nonmax_suppress_img)
save_array_as_img(nonmax_suppress_img, 'supress.jpg')

# question 5
thresholding_img = thresholding(nonmax_suppress_img, 10, 30)
show_array_as_img(thresholding_img)

