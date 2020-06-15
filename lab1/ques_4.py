import cv2
import numpy as np


def contrast_stretching_function(i, a, b, c, d):
    mid = (b - a) / (d - c)
    left = (i - c) * (mid)
    return left + a


def find_min_max__grey_pixel(img):
    return np.amin(img), np.amax(img)


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


def show_image(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)

    # destroy all windows
    cv2.destroyAllWindows()


def q4(kernal_length):
    # constants
    sigma = 1
    alpha = 1.25

    L = cv2.GaussianBlur(I, (kernal_length, kernal_length), sigma)

    H = cv2.subtract(I, L)

    _H = H * alpha
    unit_H = _H.astype('uint8')

    O = cv2.add(I, unit_H)
    # cv2.imwrite(f'{kernal_length}final_output.jpg', O)

    contrast_stretched_result = contrast_stretching(O)
    # cv2.imwrite(f'{kernal_length}contrast_stretched_result.jpg', contrast_stretched_result)


if __name__ == '__main__':

    path = 'COMP9517_20T2_Lab1_Image/cat.png'
    I = cv2.imread(path, 0)

    q4(0)