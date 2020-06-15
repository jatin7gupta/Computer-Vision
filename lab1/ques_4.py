import cv2
import numpy as np
import matplotlib.pyplot as plt
from ques_1 import read_image_grey, contrast_stretching, show_image



if __name__ == '__main__':
    path = 'COMP9517_20T2_Lab1_Image/cat.png'
    img = cv2.imread(path, 0)
    show_image(path, img)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    show_image('blur', blur)