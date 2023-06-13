import cv2
import numpy as np
import matplotlib.pylab as plt
from scipy.ndimage.filters import convolve


# apply gaussian smoothing to get rid of the noise
def gauss_smoothing(img, n, sigma):
    # initialize the filter
    gauss_filter = np.zeros((n, n), np.float32)

    # populating kernel
    n_half = n // 2
    for i in range(-n_half, n_half):
        for j in range(-n_half, n_half):
            constant = 1 / (2.0 * np.pi * sigma**2.0)
            exp = np.exp(-(i**2.0 + j**2.0) / (2 * sigma**2.0))
            gauss_filter[i+n_half, j+n_half] = constant * exp

    # convolve with the image to get smoothed image
    img = convolve(img, gauss_filter)
    return img


def image_gradient(img):
    # initialize sobel kernels
    k_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    k_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    i_x = convolve(img, k_x)
    i_y = convolve(img, k_y)

    # magnitude of gradient
    g_mag = np.hypot(i_x, i_y)
    g_mag = g_mag / g_mag.max() * 255
    theta = np.arctan2(i_y, i_x)

    return g_mag, theta


def find_threshold(img, percent_non_edge):
    t_low = 0
    t_high = 0

    hist, bin = np.histogram(img.ravel(), 256, [0, 255])
    plt.xlim([0, 255])
    plt.plot(hist)
    plt.title('histogram')
    plt.show()

    return t_low, t_high


def nonmaxima_supress(img, theta):
    m, n = img.shape
    new_img = np.zeros((m, n), dtype=np.int32)

    angle = theta * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, m - 1):
        for j in range(1, n - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    new_img[i, j] = img[i, j]
                else:
                    new_img[i, j] = 0

            except IndexError as e:
                pass

    return new_img


def edge_linking(mag_low, mag_high):

    return mag_low


def canny_detector(img, t_low, t_high):
    # smooth the image to reduce noise

    # calculate gradient

    # non maximum suppression

    # double thresholding
    return img


if __name__ == '__main__':

    input_file = 'lena.bmp'

    # read input image & show input statistics
    image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    components, counts = np.unique(image, return_counts=True)
    print(f"Input image size: {image.shape}")
    plt.imshow(image, cmap='gray')
    plt.show()

    # get smoothed image
    image = gauss_smoothing(image, 5, 1)

    # get image gradients
    gradient, theta = image_gradient(image)

    # plot histogram of the gradient and get thresholds
    find_threshold(gradient, 80)

    # view and save image
    plt.imshow(gradient, cmap='gray')
    # plt.savefig('output.png')
    plt.show()

    # suppress non maxima
    image = nonmaxima_supress(gradient, theta)

    # view and save image
    plt.imshow(cv2.convertscaleabs(image), cmap='gray')
    # plt.savefig('output.png')
    plt.show()

