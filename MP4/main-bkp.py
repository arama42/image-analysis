import cv2
import os
import numpy as np
import matplotlib.pylab as plt
from PIL import Image

def histo_equalization(img):
    # initialize a greyscale histogram(range 0-255)
    hist1 = {x: 0 for x in (range(256))}
    hist2 = {x: 0 for x in (range(256))}

    # assign count based on each pixel
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist1[img[i][j]] = hist1[img[i][j]] + 1
    print(hist1)

    # plot the input pixel value histogram
    plt.plot(hist1.keys(), hist1.values())
    plt.xlabel("Pixel intensity")
    plt.ylabel("Count of pixels")
    plt.show()

    # probability of i's
    #prob = {k: v / (img.shape[0] * img.shape[1]) for k, v in hist.items()}

    # create a cdf
    cdf = np.array(list(hist1.values())).cumsum()

    # histogram equalization
    cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf = cdf.astype('uint8')

    # get output image based on cdf
    img2 = cdf[img]

    # plot output histogram
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            hist2[img2[i][j]] = hist2[img2[i][j]] + 1

    # view and save output pixel value histogram
    plt.plot(hist2.keys(), hist2.values())
    plt.xlabel("Pixel intensity")
    plt.ylabel("Count of pixels")
    plt.show()

    # view and save output of histogram equalization
    plt.imshow(img2, cmap='Greys')
    plt.savefig('output-histogram-equalization.png')
    plt.show()

    return img2


def light_correction(img):
    A = []
    t = []

    # flatten the image
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            A.append([i, j, 1])
            t.append(img[i][j])

    A = np.array(A)
    t = np.array(t).reshape(-1, 1)

    # find the weights
    x = np.matmul(np.linalg.pinv(A), t)

    # get new intensity values
    new_intensity = np.matmul(A, x) - t

    # convert to original shape
    new_img = np.zeros(np.shape(img))
    for i in range(A.shape[0]):
        row = A[i][0]
        col = A[i][1]
        new_img[row, col] = new_intensity[i]

    # view and save image
    plt.imshow(new_img, cmap='Greys')
    plt.savefig('output-light-correction.png')
    plt.show()

    return new_img


def read_images(directory):
    images = []
    filenames = os.listdir(directory)

    for i in range(len(filenames)):
        image = cv2.imread(directory +'/'+filenames[i])
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        images.append(image_hsv)
    return images


def create_histogram(location):
    # create histogram (h,s) : frequency
    hist = {}

    # read all the dataset as HSV image list
    images = read_images(location)

    # display a sample image
    cv2.imshow('image[0]', images[0])
    cv2.waitKey(0)

    for image in images:
        # get the cropped 200 X 200 X 3 image
        n, m = image.shape[:2]
        image = image[m // 2 - 100: m // 2 + 100, n // 2 - 100: n // 2 + 100]

        # sample 4 random pixels from the image
        v, u = image.shape[:2]
        indices = np.random.randint(0, u * v, size=4)
        x, y = np.unravel_index(indices, (u, v))

        # create zero h_s
        h_s_zero = np.ndarray((2,), buffer=np.array([0, 0]), dtype=int)

        # create a histogram
        for i in range(len(x)):
            # get H & S values for the samples
            h_s = image[x[i], y[i]]
            h_s_str = str(h_s[:2])

            # add the [H,S] frequency to the histogram, discard white pixels [0, 0]
            if h_s_str in hist.keys():
                hist[h_s_str] = hist[h_s_str] + 1
            else:
                if h_s_str == str(h_s_zero):
                    continue
                else:
                    hist[h_s_str] = 1
    return hist


# Define a function to create a histogram of images
def create_histogram1(location):
    # read all the dataset as HSV image list
    images = read_images(location)

    hist = 0
    bins = 0

    for i in range(6):
        # Flatten the images into a single array of pixels
        pixels = images[i]
        print(pixels)


        hist, bins = np.histogramdd(pixels, bins=(256, 256), range=((0, 256), (0, 256)))

    return hist


if __name__ == '__main__':

    # create skin histogram
    skin_histogram = create_histogram1('Sample')
    print(len(skin_histogram))

