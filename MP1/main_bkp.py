import cv2
import numpy as np
from PIL import Image
import random

# keep track of labels
tag = 1

# hashmap for similar labels
TABLE = {}


def colourize(img):
    height, width = img.shape

    colors = []
    colors.append([])
    colors.append([])
    color = 1
    # Displaying distinct components with distinct colors
    coloured_img = Image.new("RGB", (width, height))
    coloured_data = coloured_img.load()

    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] > 0:
                if img[i][j] not in colors[0]:
                    colors[0].append(img[i][j])
                    colors[1].append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

                ind = colors[0].index(img[i][j])
                coloured_data[j, i] = colors[1][ind]

    return coloured_img


# function for getting the appropriate label based on neighbours.
def get_label(labels,  row, col):
    global tag
    global TABLE
    neighbor = 0

    upper = labels[row-1, col]
    left = labels[row, col-1]

    #print("upper: %3d left %3d" % (upper, left))

    # same, non-zero labels for neighbors
    if upper == left and upper != 0:
        neighbor = upper
    # one of the two neighbours is zero
    elif upper != left and not(upper and left):
        neighbor = max(upper, left)
    # different, non-zero labels for neighbors
    elif upper != left and upper > 0 and left > 0:
        neighbor = min(upper, left)
        other = max(upper, left)
        print("adding relation to the binary %3d - %3d" % (upper, left))

        # add relation to the dictionary
        if neighbor in TABLE.keys():
            print("a")
            TABLE[neighbor].add(other)
        elif other in TABLE.keys():
            print("b")
            TABLE[other].add(neighbor)
        else:
            flag = True
            for key in TABLE:
                if neighbor in TABLE.get(key):
                    print("c")
                    TABLE[key].add(other)
                    flag = False
                    break
                elif other in TABLE.get(key):
                    print("d")
                    TABLE[key].add(neighbor)
                    flag = False
                    break
            # create a new entry
            if flag:
                print("e")
                TABLE[neighbor] = {other}

    # if both neighbors are zero, add a new label
    else:
        neighbor = tag
        tag = tag + 1

    #print("neighbour: ", neighbor)
    return neighbor


# function for connect component labeling by iterating through the image pixels.
def ccl(path):
    global TABLE

    # read the input image
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    label = np.zeros_like(img)

    print("Shape :", img.shape)
    print("tag :", tag)
    print("table :", TABLE)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):

            if img[i][j] != 0:
                label_val = get_label(label, i, j)
                label[i][j] = label_val

    print("Shape :", img.shape)
    print("tag :", tag)
    print("table :", TABLE)
    print(label)


    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            p = label[i][j]

            if p in TABLE.keys():
                continue
            else:
                for key in TABLE:
                    if p in TABLE.get(key):
                        label[i][j] = key
                        break

    cv2.imshow('input image', label)

    backtorgb = cv2.cvtColor(label, cv2.COLOR_GRAY2RGB)
    cv2.imshow('input image', backtorgb)
    #cv2.waitKey(0)

    '''for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            p = img[i][j]

            if p in TABLE.keys():
                continue
            else:
                small_key = p
                for key in TABLE:
                    if p in TABLE.get(key) and key < small_key:
                        small_key = key
                img[i][j] = small_key

    cv2.imshow('input image', img)

    #backtorgb = cv2.cvtColor(label, cv2.COLOR_GRAY2RGB)
    #cv2.imshow('input image', backtorgb)
    #cv2.waitKey(0)

    return img'''

    return label


# main function: execute this to start the program
if __name__ == '__main__':
    face = ccl('test.bmp')
    colored_img = colourize(face)
    colored_img.show()
