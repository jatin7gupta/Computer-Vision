import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_image_grey(path):
    return cv2.imread(path, 0)


def show_image(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)

    # destroy all windows
    cv2.destroyAllWindows()


def get_list(img):
    l = []
    h = img.shape[0]
    w = img.shape[1]

    for row in range(h):
        for col in range(w):
            l.append(img[row][col])
    return l

path = 'COMP9517_20T2_Lab1_Image/cat.png'
img = read_image_grey(path)
show_image(path, img)

plt.hist(get_list(img), bins=255)
plt.ylabel('No of times')
plt.show()
