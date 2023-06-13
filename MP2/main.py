''' if you’ve got the right result. (Can you get rid of noise?) You should try different SE, different combinations and different
schemes.'''

import cv2
import math
import numpy as np


def dilation(img, kernel_size=3):
    # initialize the kernel
    kernel = np.full(shape=(kernel_size, kernel_size), fill_value=255)

    # add padding to the image
    pad_size = math.floor(kernel_size / 2 )
    img_pad = np.pad(array=img, pad_width=pad_size, mode='constant')
    print (f"\nImage size after padding: {img_pad.shape}")

    r_pad, c_pad = (img_pad.shape[0] - img.shape[0]), (img_pad.shape[1] - img.shape[1])

    # get a list of small kernel sized portions from image according to the size of the kernel.
    image_matrices = np.array([ img_pad[i:(i + kernel_size), j:(j + kernel_size)]
        for i in range(img_pad.shape[0] - r_pad) for j in range(img_pad.shape[1] - c_pad) ])

    # replace the values in the image if there are any common foreground(255) pixels overlapping with the kernel
    img_dilated = np.uint8(np.array([ 255 if (i == kernel).any() else 0 for i in image_matrices ]))
    img_dilated = img_dilated.reshape(img.shape)
    print(f"Image size after dilation: {img_dilated.shape}  {img_dilated.dtype}")

    return img_dilated

def erosion(img, kernel_size=3):
    # initialize the kernel
    kernel = np.full(shape=(kernel_size, kernel_size), fill_value=255)

    # add padding to the image
    pad_size = math.floor(kernel_size / 2 )
    img_pad = np.pad(array=img, pad_width=pad_size, mode='constant')
    print(f"\nImage size after padding: {img_pad.shape}")

    r_pad, c_pad = (img_pad.shape[0] - img.shape[0]), (img_pad.shape[1] - img.shape[1])

    # get a list of small kernel sized portions from image according to the size of the kernel.
    image_matrices = np.array([img_pad[i:(i + kernel_size), j:(j + kernel_size)]
                               for i in range(img_pad.shape[0] - r_pad) for j in range(img_pad.shape[1] - c_pad)])

    # replace the values in the image if kernel is a subset of the overlapping image portion
    img_eroded = np.uint8(np.array([255 if (i == kernel).all() else 0 for i in image_matrices]))
    img_eroded = img_eroded.reshape(img.shape)
    print(f"Image size after erosion: {img_eroded.shape}  {img_eroded.dtype}")

    return img_eroded

def opening(img, kernel_size=3):
    #a erosion b , dilation b
    img_eroded = erosion(img,kernel_size)
    image_open = dilation(img_eroded, kernel_size)
    return image_open

def closing(img, kernel_size=3):
    # a dilation b , erosion b
    img_dilated = dilation(img, kernel_size)
    image_close = erosion(img_dilated, kernel_size)
    return image_close

def boundary(img, kernel_size=3):
    # a - (a erosion b)
    img_erode = erosion(img, kernel_size)
    img_boundary = img - img_erode
    return img_boundary


if __name__ == '__main__':
    # inputs to the program
    input_file = 'gun.bmp'
    action = 'dilate'  # change to erode/open/close/boundary
    kernel_size = 7   # default, change to anything

    operations = {
        'dilate': dilation,
        'erode': erosion,
        'open': opening,
        'close': closing,
        'boundary': boundary
    }

    # read input image & show input statistics
    image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    components, counts = np.unique(image, return_counts=True)
    print(f"Input image size: {image.shape}")
    for i in range(0, len(components)):
        print(f"Component: {components[i]}, #Pixels: {counts[i]}")

    # perform action
    image_output = image
    if action in operations:
        image_output =operations[action](image, kernel_size)
    else:
        print(f"Action: {action} not available.")

    # show output statistics
    components, counts = np.unique(image_output, return_counts=True)
    print(f"\nOutput image size after {action} operation: {image_output.shape}")
    for i in range(0, len(components)):
        print(f"Component: {components[i]}, #Pixels: {counts[i]}")

    # display and save output image
    cv2.imshow(f'output image', image_output)
    cv2.waitKey(0)

    output_file = input_file.split('.')[0] + '-' + action + '.png'
    cv2.imwrite(output_file, image_output)


