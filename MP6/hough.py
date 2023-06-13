import cv2
import numpy as np
import matplotlib.pylab as plt
import os


# function for creating hough space for given image
def hough_mapping(img, rho_step=1, angle_step=1):
    m, n = img.shape

    # theta ranges from -9- to 90
    thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_step))
    print(f"num_thetas: {len(thetas)}")
    print(f"thetas: {thetas}")

    # rho ranges from -diagonal to +diagonal
    diag_len = int(np.ceil(np.sqrt(m * m + n * n)))
    rhos = np.arange(-diag_len, diag_len+1, rho_step)
    print(f"diag_len: {diag_len}")
    print(f"num_rhos: {len(rhos)}")
    print(f"rhos: {rhos}")

    # initialize hough accumulator
    h_space = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    print(f"hough space: {h_space.shape}")

    # find all edge pixel indexes
    y_idx, x_idx = np.nonzero(img)

    for i in range(len(x_idx)):
        x = x_idx[i]
        y = y_idx[i]
        for j in range(len(thetas)):
            # calculate the rho value from x, y and theta, increment the corresponding voting in h_accumulator
            rho = int((x * np.cos(thetas[j])) + (y * np.sin(thetas[j])) + diag_len)
            h_space[rho, j] += 1

    return h_space, thetas, rhos


def hough_peaks(H, num_peaks=4, level=4):
    # Create a copy of the Hough transform image
    H_copy = np.copy(H)

    # Initialize an empty list to store peak coordinates
    peaks = []

    # Loop over the desired number of peaks
    for i in range(num_peaks):
        # Find the peak in the Hough transform image
        peak = np.unravel_index(np.argmax(H_copy), H_copy.shape)
        print(f"PEAK VALUE: {peak}")
        print(f"PEAK PIXEL: {H_copy[peak]}")

        # Check if the peak is above the threshold
        if H_copy[peak] >= 0:
            # Add the peak to the list of peaks
            peaks.append(peak)

            # Zero out the neighborhood around the peak
            x, y = peak
            print(f"Y :{y}  X : {x}")
            for di in range(-1, level+1):
                for dj in range(-1, level+1):
                    neighbor_i, neighbor_j = x + di, y + dj
                    print(f"neighbor_i : {neighbor_i} neighbor_j:{neighbor_j}")
                    H_copy[neighbor_i, neighbor_j] = 0

            # set the peak to zero to avoid it from getting picked
            H_copy[peak] = 0

        else:
            # If there are no more peaks above the threshold, break out of the loop
            break

    print(f"PEAKS: {peaks}")
    # Return the list of peak coordinates
    return peaks



def hough_plot(H, folder, plot_title='Hough Space'):
    fig = plt.figure(figsize=(10, 10))
    # plt.imshow(H, cmap='gnuplot2')
    plt.imshow(H, cmap='CMRmap')
    plt.xlabel('Theta'), plt.ylabel('Rho')
    plt.tight_layout()
    plt.savefig(folder + '/' + folder + '-hough.png')
    plt.show()


def hough_lines_plot(img, indices, rhos, thetas):
    for i in range(len(indices)):
        rho = rhos[indices[i][0]]
        theta = thetas[indices[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho

        x1 = int(x0 + 1000 * -b)
        x2 = int(x0 - 1000 * -b)
        y1 = int(y0 + 1000 * a)
        y2 = int(y0 - 1000 * a)
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1)

def plot(img, hough_space, hough_lines, folder, angle_step, rho_step, level):

    # Create a figure and multiple subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (16,4))

    # Display images on each subplot in grayscale
    ax1.imshow(img)
    ax2.imshow(hough_space, cmap='CMRmap')
    ax3.imshow(hough_lines)

    # Add labels to each subplot
    ax1.set_title('Original')
    ax2.set_title('Hough Space')
    ax3.set_title('Hough lines')

    plt.suptitle('Results: angle step={}, rho step={}, level={}'.format(angle_step, rho_step, level))

    # Show the plot
    plt.savefig(folder+'-angle_step_' + str(angle_step) + "-rho_step_" + str(rho_step) + "-level_" + str(level)+".png")
    plt.plot()

if __name__ == '__main__':

    input_file = 'test2.bmp'
    angle_step = 1
    rho_step = 1
    level = 0
    num_peaks = 9

    folder = input_file.split('.')[0]
    if not os.path.exists(folder):
        os.makedirs(folder)

    # read input image
    original = cv2.imread(input_file)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    hough_line = original.copy()

    # convert image to greyscale
    image = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # detect input image edges using canny
    image = cv2.Canny(image, 50, 150, apertureSize=3)
    plt.imshow(image, cmap='gray')
    plt.savefig(folder+'/'+folder+'-canny.png')
    plt.show()

    # get hough lines accumulator
    hough_space, thetas, rhos = hough_mapping(image, rho_step, angle_step)

    # get hough peaks from hough accumulator
    indices = hough_peaks(hough_space, num_peaks, level)
    print(f"indices: {indices}")

    # plot the hough accumulator to visualize peaks
    hough_plot(hough_space, folder)
    # plot original image with hough lines
    hough_lines_plot(hough_line, indices, rhos, thetas)

    # plot everything
    plot(original, hough_space, hough_line, folder, angle_step, rho_step, level)

    # save original image with hough lines added
    plt.imshow(hough_line)
    plt.savefig(folder+'/'+folder+'-hough-lines.png')
    plt.show()
