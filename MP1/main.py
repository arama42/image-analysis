import cv2
import numpy as np
from PIL import Image
import random

# keep track of labels
current_label = 1
# hashmap for similar labels
E_TABLE = {}


def colorize(img):
    height, width = img.shape
    colors = [[], []]
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


def filter_noise(img, threshold):
    unique_labels, counts = np.unique(img, return_counts=True)
    largest_class_size = counts.max()
    size_threshold = largest_class_size * (threshold / 100)

    filtered_image = img.copy()
    for label, count in zip(unique_labels, counts):
        if count < size_threshold:
            filtered_image[filtered_image == label] = 0
    return filtered_image


# update the table to make the mappings easier
def modify_table(e_table):
    modified_table = {}
    visited = set()
    def dfs(current_label):
        visited.add(current_label)
        simplified_set = {current_label}
        for neighbor in e_table[current_label]:
            if neighbor not in visited:
                simplified_set.update(dfs(neighbor))
        return simplified_set

    for label in e_table:
        if label not in visited:
            modified_table[label] = dfs(label)

    return modified_table


# function for connect component labeling by iterating through the image pixels.
def ccl(path):
    global E_TABLE
    global current_label

    # read the input image
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    print("Initial E_TABLE :", E_TABLE)

    # first pass through the image
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):

            if image[row][col] != 0:  # check if pixel is not a background pixel
                # Find the top and left neighbors
                upper = image[row - 1, col]
                left = image[row, col - 1]

                # both neighbors are background(pixel value 0)
                if upper == 0 and left ==0:
                    image[row, col] = current_label
                    E_TABLE[current_label] = {current_label}
                    current_label = current_label+1
                # only one of the neighbors is background(pixel value 0)
                elif upper == 0 or left == 0:
                    image[row, col] = max(upper, left)
                # same, non-zero labels for neighbors
                elif upper == left and upper > 0:
                    image[row, col] = upper
                # different, non-zero labels for neighbours
                else:
                    min_val = min(upper, left)
                    max_val = max(upper, left)
                    image[row, col] = min_val
                    E_TABLE[min_val].add(max_val)
                    E_TABLE[max_val] = E_TABLE[min_val]

    updated_table = modify_table(E_TABLE)
    print("Final E_TABLE :", updated_table)

    # second pass through the image
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            component = image[row][col]

            # check the table and update to the appropriate mapping
            if component in updated_table.keys():
                continue
            else:
                for key in updated_table:
                    if component in updated_table.get(key):
                        image[row][col] = key

    return image


# main function: execute this to start the program
if __name__ == '__main__':
    input_img = 'gun.bmp'
    filter_percentage = 5

    labeled_img = ccl(input_img)
    components, counts = np.unique(labeled_img, return_counts=True)

    filtered_img = filter_noise(labeled_img, filter_percentage)
    filtered_components, filtered_counts = np.unique(filtered_img, return_counts=True)

    print("\n" + "*" * 5 + "Before filtering noise." + "*" * 5)
    for i in range(1, len(components)):
        print(f"Component: {i}, #Pixels: {counts[i]}")

    print("\n" + "*" * 5 + "After filtering noise. Threshold percent: " + str(filter_percentage) + "*" * 5)
    for i in range(1, len(filtered_components)):
        print(f"Component: {i}, #Pixels: {filtered_counts[i]}")

    colored_img = colorize(filtered_img)
    colored_img.save('gun-filtered-output.jpg')
    colored_img.show()