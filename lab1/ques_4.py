import cv2
import numpy as np


def contrast_stretching_function(i, a, b, c, d):
    mid = (b - a) / (d - c)
    left = (i - c) * (mid)
    return left + a


def find_min_max_grey_pixel(img):
    return np.amin(img), np.amax(img)


# function to implement contrast stretching
def contrast_stretching(img):
    h = img.shape[0]
    w = img.shape[1]

    a, b = 0, 255
    # let 𝑐 and 𝑑 be the minimum and maximum pixel values occurring in I
    c, d = find_min_max_grey_pixel(img)
    for row in range(h):
        for col in range(w):
            img[row][col] = contrast_stretching_function(img[row][col], a, b, c, d)
    return img


# for debugging
def show_image(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)

    # destroy all windows
    cv2.destroyAllWindows()


def q4(kernel_length):
    # constants
    sigma = 1
    alpha = 1.25

    L = cv2.GaussianBlur(I, (kernel_length, kernel_length), sigma)

    H = cv2.subtract(I, L)

    # dtype -1 signify uint8
    enhanced_H = cv2.multiply(H, alpha, dtype=-1)

    O = cv2.add(I, enhanced_H)

    cv2.imwrite(f'{kernel_length}final_output.jpg', O)

    # applying contrast stretching for more contrast between white and black
    contrast_stretched_result = contrast_stretching(O)
    cv2.imwrite(f'{kernel_length}contrast_stretched_result.jpg', contrast_stretched_result)


if __name__ == '__main__':
    path = 'COMP9517_20T2_Lab1_Image/cat.png'

    # read image as grey scale only
    I = cv2.imread(path, 0)

    # call function
    q4(3)
