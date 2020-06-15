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


if __name__ == '__main__':
    sigma = 1
    alpha = 1.25
    path = 'COMP9517_20T2_Lab1_Image/cat.png'
    I = cv2.imread(path, 0)
    cv2.imwrite('init.jpg', I)

    # show_image(path, I)

    # kernal 5,5 and sigma = 1
    L = cv2.GaussianBlur(I, (5, 5), sigma)
    # show_image('blur_image_L', L)

    H = cv2.subtract(I, L)
    # show_image('subtracted_image_H', H)

    _H = H*alpha
    unit_H = _H.astype('uint8')
    # show_image('alpha_H', H)

    O = cv2.add(I, unit_H)
    # show_image('final output', O)
    cv2.imwrite('final_output.jpg', O)

    contrast_stretched_result = contrast_stretching(O)
    # show_image('contrast_stretched_result', contrast_stretched_result)
    cv2.imwrite('contrast_stretched_result.jpg', contrast_stretched_result)