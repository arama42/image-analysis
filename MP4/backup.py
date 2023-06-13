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
    for i in range(filenames):
        image = cv2.imread(directory + filenames[i])
        images.append(image)
    return images


if __name__ == '__main__':

   ''' # read input image & show input statistics
    #image = cv2.imread(input_file)
    image = Image.open(input_file)
    image = np.array(image)

    components, counts = np.unique(image, return_counts=True)
    print(f"Input image size: {image.shape}")

    print(image[500, 700])

    # view image
    # read input image & show input statistics
    #cv2.imshow('input', image)
    #cv2.waitKey(0)

    # R channel
    plt.imshow(image[:, :, 0])
    plt.show()

    # G channel
    plt.imshow(image[:, :, 1])
    plt.show()

    # B channel
    plt.imshow(image[:, :, 2])
    plt.show()

    # R channel

    image[200:300, :, 0] = 255
    plt.imshow(image)
    plt.show()

    # G channel

    image[500:600, :, 1] = 255
    plt.imshow(image)
    plt.show()

    # B channel

    image[800:1000, :, 2] = 255
    plt.imshow(image)
    plt.show()

    # split into sub parts R G and B
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    for c, ax1 in zip(range(3), ax):
        print(f"c {c}  ax {ax1}")
        # create zero matrix
        split_img = np.zeros(image.shape, dtype="uint8")  # 'dtype' by default: 'numpy.float64'

        # adding each channel
        split_img[:, :, c] = image[:, :, c]

        # display each channel
        ax1.imshow(split_img)

    img_sample = image[500:700, 700:900]

    plt.imshow(img_sample)
    plt.show()

    colors = ("red", "green", "blue")

    # create the histogram plot, with three lines, one for
    # each color
    plt.figure()
    plt.xlim([0, 256])
    for channel_id, color in enumerate(colors):
        histogram, bin_edges = np.histogram(
            image[:, :, channel_id], bins=256, range=(0, 256)
        )
        plt.title("Color Histogram")
        plt.xlabel("Color value")
        plt.ylabel("Pixel count")
        plt.plot(bin_edges[0:-1], histogram, color=color)'''