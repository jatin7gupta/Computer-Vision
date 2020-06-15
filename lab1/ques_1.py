import cv2
import numpy as np


def read_image_grey(path):
    return cv2.imread(path, 0)


def show_image(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)

    # destroy all windows
    cv2.destroyAllWindows()


def find_min_max__grey_pixel(img):
    return np.amin(img), np.amax(img)


def contrast_stretching_function(i, a, b, c, d):
    mid = (b - a) / (d - c)
    left = (i - c) * (mid)
    return left + a


def contrast_stretching(img):
    h = img.shape[0]
    w = img.shape[1]

    a, b = 0, 255
    # let 𝑐 and 𝑑 be the minimum and maximum pixel values occurring in I
    c, d = find_min_max__grey_pixel(img)
    for row in range(h):
        for col in range(w):
            img[row][col] = contrast_stretching_function(img[row][col], a, b, c, d)
    return img


path = 'COMP9517_20T2_Lab1_Image/cat.png'

if __name__ == '__main__':
    img = read_image_grey(path)
    show_image(path, img)

    transformed_img = contrast_stretching(img)
    show_image('transformed_image', transformed_img)

