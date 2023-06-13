import cv2
import numpy as np
import matplotlib.pylab as plt


def histo_equalization(img):
    # initialize a greyscale histogram(range 0-255)
    hist1 = {x: 0 for x in (range(256))}
    hist2 = {x: 0 for x in (range(256))}

    # assign count based on each pixel
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist1[img[i][j]] = hist1[img[i][j]] + 1
    print(hist1)

    # plot the histogram of pixel values
    plt.plot(hist1.keys(), hist1.values())
    plt.xlabel("Pixel intensity")
    plt.ylabel("Count of pixels")
    plt.show()

    # probability of i's
    #prob = {k: v / (img.shape[0] * img.shape[1]) for k, v in hist.items()}

    # create a cdf
    cdf = np.array(list(hist1.values())).cumsum()
    print(cdf)

    # normalize cdf
    # cdf_normalized = cdf * max(hist.values()) / cdf.max()
    # print(cdf_normalized)

    # histogram equalization
    # cdf_m = np.ma.masked_equal(cdf, 0)
    cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf = cdf.astype('uint8')
    # cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    # get output image based on cdf
    img2 = cdf[img]

    # plot output histogram
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            hist2[img2[i][j]] = hist2[img2[i][j]] + 1
    print(hist2)
    plt.plot(hist2.keys(), hist2.values())
    plt.xlabel("Pixel intensity")
    plt.ylabel("Count of pixels")
    plt.show()

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


if __name__ == '__main__':
    input_file = 'moon.bmp'

    # read input image & show input statistics
    image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)

    components, counts = np.unique(image, return_counts=True)
    print(f"Input image size: {image.shape}")

    #for i in range(0, len(components)):
    #    print(f"Component: {components[i]}, #Pixels: {counts[i]}")
    #hist_numpy = dict(zip(components, counts))
    #print(hist_numpy)

    # histogram equalization of the image
    output_histogram = histo_equalization(image)

    # get light corrected image
    output_corrected = light_correction(output_histogram)

    # display and save histogram image
    cv2.imshow(f'output-histogram.png', output_histogram)
    cv2.waitKey(0)
    cv2.imwrite('output-histogram.png', output_histogram)

    # display and save output image
    cv2.imshow(f'output-corrected.png', output_corrected)
    cv2.waitKey(0)
    cv2.imwrite('output-corrected.png', output_corrected)
