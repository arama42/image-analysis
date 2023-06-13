import cv2
import os
import numpy as np
import matplotlib.pylab as plt


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
                segmented[i][j][2] = pixel[2]
                print(pixel[0], pixel[1], pixel[2])
                print(segmented[i][j])

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
    plt.imshow(segmented, cmap='gray')
    plt.savefig('output-segmented.png')
    plt.show()

    cv2.imshow('output', segmented)
    cv2.waitKey(0)






