import cv2
import numpy as np
import matplotlib.pyplot as plt
from ques_1 import read_image_grey, path, contrast_stretching, show_image


def convolution(img, kernel):

    img_row, img_col = img.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(img.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((img_row + (2 * pad_height), img_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = img

    for row in range(img_row):
        for col in range(img_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])

    return output


def normalize(img):
    h = img.shape[0]
    w = img.shape[1]

    for row in range(h):
        for col in range(w):
            pixel_val = img[row][col]
            if pixel_val < 0:
                img[row][col] = 0
            elif pixel_val > 255:
                img[row][col] = 255
            else:
                img[row][col] = int(pixel_val)
    return img


def sobel_edge_detection(image):
    filter_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])

    new_image_x = convolution(image, filter_x)

    filter_y = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]
                         ])
    new_image_y = convolution(image, filter_y)

    return new_image_x, new_image_y


if __name__ == '__main__':
    # read normal image
    img = read_image_grey(path)
    show_image(path, img)

    # sobel edge detection
    edge_detected_x, edge_detected_y = sobel_edge_detection(img)
    edge_detected_x = normalize(edge_detected_x)
    edge_detected_y = normalize(edge_detected_y)
    show_image('new_image_x', edge_detected_x)
    show_image('new_image_y', edge_detected_y)

    # sobel_y = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=5)
    # sobel_x = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=5)

    # show_image('inbuilt_sobel_x', sobel_x)
    # show_image('inbuilt_sobel_y', sobel_y)



