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
    h_hist = {x: 0 for x in (range(256))}
    s_hist = {x: 0 for x in (range(256))}

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
            h = h_s[0]
            s = h_s[1]

            # add the [H,S] frequency to the histogram, discard white pixels [0, 0]
            if h == 0:
                continue
            else:
                h_hist[h] = h_hist[h] + 1
            # add the [H,S] frequency to the histogram, discard white pixels [0, 0]
            if s == 0:
                continue
            else:
                s_hist[s] = s_hist[s] + 1

    return h_hist, s_hist

def segment_hsv(image, h_low, h_high, s_low, s_high):
    segmented = np.zeros(np.shape(image))

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    for i in range(image_hsv.shape[0]):
        for j in range(image_hsv.shape[1]):
            pixel = image_hsv[i][j]
            if (h_low <= pixel[0] <= h_high) and (s_low <= pixel[1] <= s_high):
                segmented[i][j][0] = pixel[0]
                segmented[i][j][1] = pixel[1]
                segmented[i][j][1] = pixel[2]

    return segmented


if __name__ == '__main__':

    # create skin histogram
    '''skin_hist_h, skin_hist_s = create_histogram('Hands')
    print(skin_hist_h)
    print(skin_hist_s)
    
    # plot histogram
    #plt.hist2d(skin_hist_h.values(),skin_hist_s.values())
    plt.plot(skin_hist_h.keys(), skin_hist_h.values(), label='H')
    plt.plot(skin_hist_s.keys(), skin_hist_s.values(), label='S')
    plt.xlabel("Pixel intensity")
    plt.ylabel("Count of pixels")
    plt.show()'''

    image = cv2.imread('pointer1.bmp')
    cv2.imshow('image', image)
    cv2.waitKey(0)
    segmented = segment_hsv(image, 0, 25, 50, 150)


    #segmented = cv2.cvtColor(segmented, cv2.COLOR_HSV2RGB)
    # view and save image
    plt.imshow(segmented, cmap='Greys')
    plt.savefig('output-segmented.png')
    plt.show()

    cv2.imshow('segmented', segmented)
    cv2.waitKey(0)



