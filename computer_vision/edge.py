import matplotlib.pyplot as plt
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
    vertical_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
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
    theta = np.arctan2(Gy, Gx) * 180 / np.pi
    theta[theta < 0] += 180
    x, y = G.shape
    suppressed_G = np.zeros(G.shape)

    for i in range(1, x - 1):
        for j in range(1, y - 1):
            if (0 <= theta[i, j] < 22.5) or (157.5 <= theta[i, j] <= 180):
                N1 = G[i, j + 1]
                N2 = G[i, j - 1]
            elif 22.5 <= theta[i, j] < 67.5:
                N1 = G[i - 1, j - 1]
                N2 = G[i + 1, j + 1]
            elif 67.5 <= theta[i, j] < 112.5:
                N1 = G[i + 1, j]
                N2 = G[i - 1, j]
            else:
                N1 = G[i + 1, j - 1]
                N2 = G[i - 1, j + 1]

            if (G[i, j] >= N1) and (G[i, j] >= N2):
                suppressed_G[i, j] = G[i, j]
    return suppressed_G

def thresholding(G, low, high):
    '''Binarize G according threshold low and high'''
    # TODO: Please complete this function.
    # your code here
    assert high <= 361
    assert low >= 0 & low < high
    x, y = G.shape
    result = np.zeros(G.shape)
    weak = 10
    strong = 255
    G_low = G.copy()
    G_high = G.copy()
    G_low[G_low < low] = 0
    G_low[G_low >= high] = 0
    G_low[G_low > 0] = 255
    G_high[G_high < high] = 0
    G_high[G_high > 0] = 255
    # show_array_as_img(G_low)
    save_array_as_img(G_low, 'edgemap_low.jpg')
    # show_array_as_img(G_high)
    save_array_as_img(G_high, 'edgemap_high.jpg')

    strong_i, strong_j = np.where(G >= high)
    weak_i, weak_j = np.where((G < high) & (G >= low))
    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak
    for i in range(1, x-1):
        for j in range(1, y-1):
            if result[i,j] == weak:
                if (result[i+1, j-1] == strong or result[i+1, j] == strong or result[i+1, j+1] == strong
                    or result[i, j-1] == strong or result[i, j+1] == strong or result[i-1, j-1] == strong
                        or result[i-1, j] == strong or result[i-1, j+1] == strong):
                    result[i, j] = strong
                else:
                    result[i, j] = 0
    G = result
    return G

def hough(G):
    '''Return Hough transform of G'''
    # TODO: Please complete this function.
    # your code here
    steps = 10
    x, y = G.shape
    D = int(np.sqrt(np.square(x) + np.square(y)))
    thetas = np.deg2rad(np.arange(0, 180, step=180/steps))
    rhos = np.linspace(-D, D, 2*D)
    accumulator = np.zeros((len(rhos), len(thetas)))
    for i in range(x):
        for j in range(y):
            if G[i, j] == 255:
                for k in range(len(thetas)):
                    rho = j * np.cos(thetas[k]) + i * np.sin(thetas[k])
                    accumulator[int(rho) + D, k] += 1
    return accumulator, thetas, rhos


img = read_img_as_array('road.jpeg')
#TODO: detect edges on 'img'
# question 1
gray_img = rgb2gray(img)
# show_array_as_img(gray_img)
save_array_as_img(gray_img, 'gray.jpg')

# question 2
gauss_img = ndimage.gaussian_filter(gray_img, sigma=1.2)
# show_array_as_img(gauss_img)
save_array_as_img(gauss_img, 'gauss.jpg')

# question 3
G, Gx, Gy = sobel(gauss_img)
# show_array_as_img(G)
# show_array_as_img(Gx)
# show_array_as_img(Gy)
save_array_as_img(G, 'G.jpg')
save_array_as_img(Gx, 'G_x.jpg')
save_array_as_img(Gy, 'G_y.jpg')

# question 4
nonmax_suppress_img = nonmax_suppress(G,Gx,Gy)
# show_array_as_img(nonmax_suppress_img)
save_array_as_img(nonmax_suppress_img, 'supress.jpg')

# question 5
thresholding_img = thresholding(nonmax_suppress_img, 20, 40)
# show_array_as_img(thresholding_img)
save_array_as_img(thresholding_img, 'edgemap.jpg')
accumulator, thetas, rhos = hough(thresholding_img)
show_array_as_img(accumulator)
# idx = np.argmax(accumulator)
# rho = int(rhos[int(idx / accumulator.shape[1])])
# theta = thetas[int(idx % accumulator.shape[1])]
# print("rho={0:.0f}, theta={1:.0f}".format(rho, np.rad2deg(theta)))
print(accumulator.shape)
