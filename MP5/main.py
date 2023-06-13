import cv2
import os
import numpy as np
import matplotlib.pylab as plt


# convolution of matrix a to f
def conv2d(a, f):
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    stride = np.lib.stride_tricks.as_strided
    sub_m = stride(a, shape=s, strides=a.strides * 2)
    return np.einsum('ij,ijkl->kl', f, sub_m)


# apply gaussian smoothing to get rid of the noise from image
def gauss_smoothing(img, n, sigma):
    # initialize the filter
    g_filter = np.zeros((n, n), np.float32)

    # populating kernel
    n_half = n // 2
    for i in range(-n_half, n_half):
        for j in range(-n_half, n_half):
            constant = 1 / (2.0 * np.pi * sigma**2.0)
            exp = np.exp(-(i**2.0 + j**2.0) / (2 * sigma**2.0))
            g_filter[i+n_half, j+n_half] = constant * exp

    # convolve with the image to get smoothed image
    img = conv2d(img, g_filter)
    return img


def image_gradient(img, detector='sobel'):
    if detector == 'sobel':
        # initialize sobel kernels
        k_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        k_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    elif detector == 'roberts':
        k_x = np.array([[1, 0], [0, -1]], np.float32)
        k_y = np.array([[0, -1], [1, 0]], np.float32)
    elif detector == 'prewitt':
        k_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)
        k_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], np.float32)
    else:
        print(f"Not a valid filter")

    # convolve image with kernels to get gradients
    i_x = conv2d(img, k_x)
    i_y = conv2d(img, k_y)

    # magnitude of gradient
    g_mag = np.hypot(i_x, i_y)
    g_mag = g_mag / g_mag.max() * 255
    theta = np.arctan2(i_y, i_x)

    return g_mag, theta


def find_threshold(img, pt_non_edge, n, output_folder):
    # Calculate the histogram of the image gradient
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])

    # plot histogram of the gradient
    plt.hist(img.flatten(), bins=256, range=[0, 256])
    plt.xlabel('Intensity')
    plt.ylabel('Count')
    plt.title('Gradient Histogram')
    plt.savefig(output_folder+'/n_'+str(n)+'-pne_'+str(pt_non_edge)+'-histogram.png')
    plt.show()

    # create a cdf of the pixels
    cdf = hist.cumsum()

    # normalize cdf
    cdf_norm = cdf / np.max(cdf)

    # Plot the CDF
    plt.plot(bins[:-1], cdf_norm)
    plt.xlabel('Intensity')
    plt.ylabel('Cumulative Probability')
    plt.title('CDF')
    plt.savefig(output_folder + '/n_' + str(n) + '-pne_' + str(pt_non_edge) + '-cdf.png')
    plt.show()

    indices = np.where(cdf_norm > pt_non_edge)
    intensities = bins[:-1][indices]

    t_high = int(intensities.min())
    t_low = 0.5 * t_high
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


# get strong and weak pixels
def threshold(img, low, high):
    m, n = img.shape
    res = np.zeros((m, n), dtype=np.int32)

    weak = np.int32(100)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= high)
    zeros_i, zeros_j = np.where(img < low)
    weak_i, weak_j = np.where((img <= high) & (img >= low))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res


def edge_linking_recursion(img):

    m, n = img.shape
    visited = set()

    for i in range(1, m - 1):
        for j in range(1, n - 1):
            if img[i, j] == 255 and (i, j) not in visited:
                # Follow the edge from the current pixel
                edge_list = follow_edges(i, j, img, visited)

                # If the edge is longer than a threshold, mark it as an edge
                if len(edge_list) >= 10:
                    for (edge_i, edge_j) in edge_list:
                        img[edge_i, edge_j] = 255

    # Set all non-edge pixels to zero
    img[img < 255] = 0
    return img


def follow_edges(i, j, img, visited):
    m, n = img.shape
    edge_list = []

    for di in range(-1, 2):
        for dj in range(-1, 2):
            neighbor_i, neighbor_j = i + di, j + dj

            # Skip pixels outside the image boundary
            if neighbor_i < 0 or neighbor_i >= m or neighbor_j < 0 or neighbor_j >= n:
                continue

            # Skip pixels already visited or below the low threshold
            if (neighbor_i, neighbor_j) in visited or img[neighbor_i, neighbor_j] < 100:
                continue

            # Add the current pixel to the edge list and mark it as visited
            edge_list.append((neighbor_i, neighbor_j))
            visited.add((neighbor_i, neighbor_j))

            # Recursively follow the edge from the current pixel
            edge_list.extend(follow_edges(neighbor_i, neighbor_j, img, visited))

    return edge_list


def plot(img_gauss, img_grad, img_nms, img_canny, N, sigma, detector, pne, output_folder):

    # Create a figure and multiple subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize = (16,4))

    # Display images on each subplot in grayscale
    ax1.imshow(img_gauss, cmap='gray')
    ax2.imshow(img_grad, cmap='gray')
    ax3.imshow(img_nms, cmap='gray')
    ax4.imshow(img_canny, cmap='gray')

    # Add labels to each subplot
    ax1.set_title('Gaussian Smoothed')
    ax2.set_title('Gradient')
    ax3.set_title('NMS')
    ax4.set_title('Canny Edge')

    plt.suptitle('Results: Filter size={}, Sigma={}, Percentage of non-edges={}, Filter={}'.format(N, sigma, pne, detector))

    # Show the plot
    output = 'output'
    plt.savefig(output+'/'+output_folder+'-filter_size_' + str(N) + "sigma_val_" + str(sigma) + "percentage_of_non_edge_" + str(pne)+".png")
    plt.plot()


if __name__ == '__main__':

    input_files = ['lena.bmp', 'joy1.bmp', 'pointer1.bmp', 'test1.bmp']
    detectors = ['sobel', 'roberts', 'prewitt']
    ns = [3, 5, 7]
    sigmas = [1, 2, 3]
    pt_non_edges = [0.80, 0.85, 0.90]

    for input_file in input_files:
        for detector in detectors:
            for sigma in sigmas:
                # create output folder
                input_name = input_file.split('.')[0]
                output_folder = input_name+'-'+detector+'-'+str(sigma)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                for n in ns:
                    for pt_non_edge in pt_non_edges:

                        # read input image & show input statistics
                        image = cv2.imread(input_file)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                        # get gaussian smoothed image
                        image_gauss = gauss_smoothing(image, n, sigma)

                        # view & save image
                        plt.imshow(image_gauss, cmap='gray')
                        plt.savefig(output_folder+'/n_'+str(n)+'-pne_'+str(pt_non_edge)+'-gaussian-smooth.png')
                        plt.show()

                        # get image gradients
                        image_grad, theta = image_gradient(image_gauss, detector)

                        # view & save image
                        plt.imshow(image_grad, cmap='gray')
                        plt.savefig(output_folder+'/n_'+str(n)+'-pne_'+str(pt_non_edge)+'-gradient.png')
                        plt.show()

                        # plot histogram of the gradient and get thresholds
                        t_low, t_high = find_threshold(image_grad, pt_non_edge, n, output_folder)
                        print(f"t_low {t_low} t_high {t_high}")

                        # suppress non maxima
                        image_nms = nonmaxima_supress(image_grad, theta)

                        # view and save image
                        plt.imshow(cv2.convertScaleAbs(image_nms), cmap='gray')
                        plt.savefig(output_folder+'/n_'+str(n)+'-pne_'+str(pt_non_edge)+'-nms.png')
                        plt.show()

                        # threshold
                        image_thr = threshold(image_nms, t_low, t_high)

                        # view and save image
                        plt.imshow(cv2.convertScaleAbs(image_thr), cmap='gray')
                        plt.savefig(output_folder+'/n_'+str(n)+'-pne_'+str(pt_non_edge)+'-threshold.png')
                        plt.show()

                        # edge linking
                        image_canny = edge_linking_recursion(image_thr)

                        # view and save image
                        plt.imshow(cv2.convertScaleAbs(image_canny), cmap='gray')
                        plt.savefig(output_folder+'/n_'+str(n)+'-pne_'+str(pt_non_edge)+'-edge-linking.png')
                        plt.show()

                        plot(image_gauss, image_grad, image_nms, image_canny, n, sigma, detector, pt_non_edge, output_folder)

