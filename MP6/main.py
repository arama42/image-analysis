import cv2
import numpy as np
import matplotlib.pylab as plt


# 2D convolution
def conv2d(a, f):
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    stride = np.lib.stride_tricks.as_strided
    sub_m = stride(a, shape=s, strides=a.strides * 2)
    return np.einsum('ij,ijkl->kl', f, sub_m)


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
    img = conv2d(img, gauss_filter)
    return img


def image_gradient(img):
    # initialize sobel kernels
    k_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    k_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    # convolve image with kernels to get gradients
    i_x = conv2d(img, k_x)
    i_y = conv2d(img, k_y)

    # magnitude of gradient
    g_mag = np.hypot(i_x, i_y)
    g_mag = g_mag / g_mag.max() * 255
    theta = np.arctan2(i_y, i_x)

    return g_mag, theta


def find_threshold(img, percent_non_edge):
    t_low = 0
    t_high = 0

    plt.hist(img.ravel(), 256, [0, 255])
    plt.title('histogram')
    plt.savefig('gradient-histogram.png')
    plt.show()

    cdf = np.array(img.flatten()).cumsum()
    print(cdf)

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


def threshold(img, high, low):
    highThreshold = img.max() * high;
    lowThreshold = highThreshold * low;

    m, n = img.shape
    res = np.zeros((m,n), dtype=np.int32)

    weak = np.int32(100)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res


def edge_inking(img):
    m, n = img.shape
    weak = 100
    strong = 255

    top_to_bottom = img.copy()

    for row in range(1, m):
        for col in range(1, n):
            if top_to_bottom[row, col] == weak:
                if top_to_bottom[row, col + 1] == 255 or top_to_bottom[row, col - 1] == 255 or top_to_bottom[
                    row - 1, col] == 255 or top_to_bottom[
                    row + 1, col] == 255 or top_to_bottom[
                    row - 1, col - 1] == 255 or top_to_bottom[row + 1, col - 1] == 255 or top_to_bottom[
                        row - 1, col + 1] == 255 or top_to_bottom[row + 1, col + 1] == 255:
                    top_to_bottom[row, col] = 255
                else:
                    top_to_bottom[row, col] = 0

    bottom_to_top = img.copy()

    for row in range(m - 1, 0, -1):
        for col in range(n - 1, 0, -1):
            if bottom_to_top[row, col] == weak:
                if bottom_to_top[row, col + 1] == 255 or bottom_to_top[row, col - 1] == 255 or bottom_to_top[
                    row - 1, col] == 255 or bottom_to_top[
                    row + 1, col] == 255 or bottom_to_top[
                    row - 1, col - 1] == 255 or bottom_to_top[row + 1, col - 1] == 255 or bottom_to_top[
                    row - 1, col + 1] == 255 or bottom_to_top[
                        row + 1, col + 1] == 255:
                    bottom_to_top[row, col] = 255
                else:
                    bottom_to_top[row, col] = 0

    right_to_left = img.copy()

    for row in range(1, m):
        for col in range(n - 1, 0, -1):
            if right_to_left[row, col] == weak:
                if right_to_left[row, col + 1] == 255 or right_to_left[row, col - 1] == 255 or right_to_left[
                    row - 1, col] == 255 or right_to_left[
                    row + 1, col] == 255 or right_to_left[
                    row - 1, col - 1] == 255 or right_to_left[row + 1, col - 1] == 255 or right_to_left[
                    row - 1, col + 1] == 255 or right_to_left[
                        row + 1, col + 1] == 255:
                    right_to_left[row, col] = 255
                else:
                    right_to_left[row, col] = 0

    left_to_right = img.copy()

    for row in range(m - 1, 0, -1):
        for col in range(1, n):
            if left_to_right[row, col] == weak:
                if left_to_right[row, col + 1] == 255 or left_to_right[row, col - 1] == 255 or left_to_right[
                    row - 1, col] == 255 or left_to_right[
                    row + 1, col] == 255 or left_to_right[
                    row - 1, col - 1] == 255 or left_to_right[row + 1, col - 1] == 255 or left_to_right[
                    row - 1, col + 1] == 255 or left_to_right[
                        row + 1, col + 1] == 255:
                    left_to_right[row, col] = 255
                else:
                    left_to_right[row, col] = 0

    final_image = top_to_bottom + bottom_to_top + right_to_left + left_to_right

    final_image[final_image > 255] = 255

    return final_image


if __name__ == '__main__':

    input_file = 'pointer1.bmp'

    # read input image & show input statistics
    image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    components, counts = np.unique(image, return_counts=True)
    print(f"Input image size: {image.shape}")
    plt.imshow(image, cmap='gray')
    plt.show()

    # get gaussian smoothed image
    image = gauss_smoothing(image, 5, 1)

    # view & save image
    plt.imshow(image, cmap='gray')
    plt.savefig(input_file.split('.')[0]+'-gaussian-smooth.png')
    plt.show()

    # get image gradients
    gradient, theta = image_gradient(image)

    # plot histogram of the gradient and get thresholds
    find_threshold(gradient, 80)

    # view & save image
    plt.imshow(gradient, cmap='gray')
    plt.savefig(input_file.split('.')[0] + '-sobel-gradient.png')
    plt.show()

    # suppress non maxima
    image = nonmaxima_supress(gradient, theta)

    # view and save image
    plt.imshow(cv2.convertScaleAbs(image), cmap='gray')
    plt.savefig(input_file.split('.')[0] + '-non-maxima-suppression.png')
    plt.show()

    # threshold
    image = threshold(image, 0.25, 0.45)

    # view and save image
    plt.imshow(cv2.convertScaleAbs(image), cmap='gray')
    plt.savefig(input_file.split('.')[0] + '-threshold.png')
    plt.show()

    # edge linking
    image = edge_inking(image)

    # view and save image
    plt.imshow(cv2.convertScaleAbs(image), cmap='gray')
    plt.savefig(input_file.split('.')[0] + '-edge-linking.png')
    plt.show()

