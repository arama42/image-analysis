import cv2
import math
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

# algo
# map each point (x,y) to lines in m,c space
# set of colinear points in x-y space is mapped to set of lines intersecting in m-c space
# intersection of the line in m, c space is m0, c0

#
# ∀(x,y) ∈ E
# Draw a line c=-xm+y in the m-c plane
# if the line passes through (m,c), A[m,c] ++

# Find local maxima of A Æ lines in E

# different settings for the quantization of the parameter space
# idea of detecting those significant intersections


# function for creating hough accumulator for given image
def hough_accumulator(img, rho_step=1, angle_step=1, theta_low=-90.0, theta_high=90.0, rho_range=0):
    m, n = img.shape

    # theta ranges from -9- to 90
    thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_step))
    print(f"num_thetas: {len(thetas)}")
    # print(f"thetas: {thetas}")

    # rho ranges from -diagonal to +diagonal
    diag_len = int(np.ceil(np.sqrt(m * m + n * n)))
    rhos = np.arange(-diag_len, diag_len+1, rho_step)
    print(f"diag_len: {diag_len}")
    print(f"num_rhos: {len(rhos)}")
    # print(f"rhos: {rhos}")

    # initialize hough accumulator
    h_accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    print(f"hough accumulator: {h_accumulator.shape}")

    # find all edge pixel indexes
    y_idx, x_idx = np.nonzero(img)

    for i in range(len(x_idx)):
        x = x_idx[i]
        y = y_idx[i]

        for j in range(len(thetas)):
            # calculate the rho value from x, y and theta, increment the corresponding voting in h_accumulator
            rho = int((x * np.cos(thetas[j])) + (y * np.sin(thetas[j])) + diag_len)
            h_accumulator[rho, j] += 1

    return h_accumulator, thetas, rhos

# get hough peaks
def hough_peaks(H, num_peaks, threshold=0, nhood_size=3):
    ''' A function that returns the indicies of the accumulator array H that
        correspond to a local maxima.  If threshold is active all values less
        than this value will be ignored, if neighborhood_size is greater than
        (1, 1) this number of indicies around the maximum will be surpessed. '''
    # loop through number of peaks to identify
    indicies = []
    H1 = np.copy(H)
    for i in range(num_peaks):
        idx = np.argmax(H1) # find argmax in flattened array
        H1_idx = np.unravel_index(idx, H1.shape) # remap to shape of H
        indicies.append(H1_idx)

        # surpess indicies in neighborhood
        idx_y, idx_x = H1_idx # first separate x, y indexes from argmax(H)
        # if idx_x is too close to the edges choose appropriate values
        if (idx_x - (nhood_size/2)) < 0: min_x = 0
        else: min_x = idx_x - (nhood_size/2)
        if ((idx_x + (nhood_size/2) + 1) > H.shape[1]): max_x = H.shape[1]
        else: max_x = idx_x + (nhood_size/2) + 1

        # if idx_y is too close to the edges choose appropriate values
        if (idx_y - (nhood_size/2)) < 0: min_y = 0
        else: min_y = idx_y - (nhood_size/2)
        if ((idx_y + (nhood_size/2) + 1) > H.shape[0]): max_y = H.shape[0]
        else: max_y = idx_y + (nhood_size/2) + 1

        # bound each index by the neighborhood size and set all values to 0
        min_x = int(min_x)
        max_x = int(max_x)
        min_y = int(min_y)
        max_y = int(max_y)

        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                # remove neighborhoods in H1
                H1[y, x] = 0

                # highlight peaks in original H
                if (x == min_x or x == (max_x - 1)):
                    H[y, x] = 255
                if (y == min_y or y == (max_y - 1)):
                    H[y, x] = 255

    # return the indicies and the original Hough space with selected points
    return indicies, H


def find_hough_peaks(h_accumulator, num_peaks, threshold=0):

    # TODO
    hist, bins = np.histogram(h_accumulator, density=True)
    plt.hist(h_accumulator, bins=bins)
    print(f"bins: {bins}")
    plt.title('histogram')
    plt.show()

    # heatmap of accumulator
    '''plt.imshow(h_accumulator, cmap='hot', interpolation='nearest')
    plt.title('heatmap')
    plt.show()'''

    ax = sns.heatmap(h_accumulator, linewidth=0.5)
    plt.show()

    # Flatten the array and get the indices of the top `num_peaks` values
    flat_idx = np.argpartition(h_accumulator.flatten(), -num_peaks)[-num_peaks:]
    print(f"flat_idx: {flat_idx}")
    rows, cols = np.unravel_index(flat_idx, h_accumulator.shape)
    print(f"rows: {rows} cols {cols}")

    # Filter out peaks that are below the threshold
    mask = h_accumulator[rows, cols] > threshold
    rows = rows[mask]
    cols = cols[mask]

    # Sort the peaks by their accumulator values (in descending order)
    indices_sorted = np.argsort(-h_accumulator[rows, cols])
    rows = rows[indices_sorted]
    cols = cols[indices_sorted]

    # Return a list of tuples of (row, col) indices
    return [(r, c) for r, c in zip(rows, cols)]


def find_hough_peaks_new(h_accumulator, ratio=0.5):
    # TODO
    hist, bins = np.histogram(h_accumulator, density=True)
    plt.hist(h_accumulator, bins=bins)
    print(f"bins: {bins}")
    plt.title('histogram')
    plt.show()

    # heatmap of accumulator
    plt.imshow(h_accumulator, cmap='hot', interpolation='nearest')
    plt.title('heatmap')
    plt.show()

    ax = sns.heatmap(h_accumulator, linewidth=0.5)
    plt.show()

    # Flatten the array and get the indices of the peaks
    threshold = ratio * np.max(h_accumulator)
    flat_idx = np.argwhere(h_accumulator > threshold)
    print(f"flat_idx: {flat_idx}")
    rows, cols = flat_idx[:,0], flat_idx[:,1]
    print(f"rows: {rows} cols {cols}")

    # Sort the peaks by their accumulator values (in descending order)
    indices_sorted = np.argsort(-h_accumulator[rows, cols])
    rows = rows[indices_sorted]
    cols = cols[indices_sorted]

    # Return a list of tuples of (row, col) indices
    return [(r, c) for r, c in zip(rows, cols)]


def plot_hough_acc(H, plot_title='Hough Accumulator Plot'):
    ''' A function that plot a Hough Space using Matplotlib. '''
    fig = plt.figure(figsize=(10, 10))

    plt.imshow(H, cmap='jet')

    plt.xlabel('Theta Direction'), plt.ylabel('Rho Direction')
    plt.tight_layout()
    plt.show()


def hough_lines_draw(img, indicies, rhos, thetas):
    ''' A function that takes indicies a rhos table and thetas table and draws
        lines on the input images that correspond to these values. '''
    for i in range(len(indicies)):
        # reverse engineer lines from rhos and thetas
        rho = rhos[indicies[i][0]]
        theta = thetas[indicies[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        # these are then scaled so that the lines go off the edges of the image
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


if __name__ == '__main__':
    input_file = 'input.bmp'
    folder = input_file.split('.')[0]

    # read input image
    original = cv2.imread(input_file)

    # convert image to greyscale
    image = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # detect input image edges
    image = cv2.Canny(image, 50, 150, apertureSize=3)
    plt.imshow(image, cmap='gray')
    plt.savefig(folder+'/'+folder+'-edge.png')
    plt.show()

    # get hough lines accumulator
    accumulator, thetas, rhos = hough_accumulator(image)

    # get hough peaks from hough accumulator
    indices = find_hough_peaks_new(accumulator, 0.5)
    print(f"indices: {indices}")

    plot_hough_acc(accumulator)

    hough_lines_draw(original, indices, rhos, thetas)

    # Show image with manual Hough Transform Lines
    cv2.imshow('Major Lines: Manual Hough Transform', original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()






