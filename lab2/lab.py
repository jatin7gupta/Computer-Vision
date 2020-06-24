import imutils
# Task1 Hint: (with sample code for the SIFT detector)
# Initialize SIFT detector, detect keypoints, store and show SIFT keypoints of original image in a Numpy array
# Define parameters for SIFT initializations such that we find only 10% of keypoints
import cv2
import matplotlib.pyplot as plt


class SiftDetector():
    def __init__(self, norm="L2", params=None):
        self.detector=self.get_detector(params)
        self.norm=norm

    def get_detector(self, params):
        if params is None:
            params={}
            params["n_features"]=0
            params["n_octave_layers"]=3
            params["contrast_threshold"]=0.03
            params["edge_threshold"]=10
            params["sigma"]=1.6

        detector = cv2.xfeatures2d.SIFT_create(
                nfeatures=params["n_features"],
                nOctaveLayers=params["n_octave_layers"],
                contrastThreshold=params["contrast_threshold"],
                edgeThreshold=params["edge_threshold"],
                sigma=params["sigma"])

        return detector



# Task2 Hint:
# Upscale the image, compute SIFT features for rescaled image
# Apply BFMatcher with defined params and ratio test to obtain good matches, and then select and draw best 5 matches

# Task3 Hint: (with sampe code for the rotation)
# Rotate the image and compute SIFT features for rotated image
# Apply BFMatcher with defined params and ratio test to obtain good matches, and then select and draw best 5 matches
import math
import numpy as np
import sys

# image: image to rotate
# x:     x-coordinate of point we wish to rotate around
# y:     y-coordinate of point we wish to rotate around
# angle: degrees to rotate image by
#
# Returns a rotated copy of the original image
def rotate(image, x, y, angle):
    rot_matrix = cv2.getRotationMatrix2D((x, y), angle, 1.0)
    h, w = image.shape[:2]

    return cv2.warpAffine(image, rot_matrix, (w, h))

# Get coordinates of center point.
#
# image:  Image that will be rotated
# return: (x, y) coordinates of point at center of image
def get_img_center(image):
    height, width = image.shape[:2]
    center = height // 2, width // 2

    return center


img = cv2.imread('COMP9517_20T2_Lab2_Image.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def task1_a(img, name):
    sift = SiftDetector()
    detector = sift.get_detector(None)
    
    kp = detector.detect(img, None)
    img = cv2.drawKeypoints(img,kp,img)
    cv2.imwrite(name, img)


# task1_a(img.copy(), 'task1_A.jpg')


def task1_b(img, name):
    
    detector = get_sift_detector()

    kp = detector.detect(img, None)
    img = cv2.drawKeypoints(img, kp, img)
    cv2.imwrite(name, img)
    return img


def get_sift_detector():
    sift = SiftDetector()
    params = {}
    params["n_features"] = 0
    params["n_octave_layers"] = 3
    params["contrast_threshold"] = 0.136
    params["edge_threshold"] = 10
    params["sigma"] = 1.6
    detector = sift.get_detector(params)
    return detector


# task1_b(img.copy(), 'task1_B.jpg')

def task1_b_n(img, name):


    sift = SiftDetector()
    
    params={}
    params["n_features"]=620
    params["n_octave_layers"]=3
    params["contrast_threshold"]=0.03
    params["edge_threshold"]=10
    params["sigma"]=1.6

    detector = sift.get_detector(params)
    
    kp = detector.detect(img, None)
    img = cv2.drawKeypoints(img,kp, img)
    cv2.imwrite(name, img)
# task1_b_n(img.copy(), 'task1_B_n.jpg')


def task2(img, name):
    # part a
    scale_percent = 115  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim)

    # part b: increased to 710 from 625
    resized_image = task1_b(resized, name + '_B.jpg')

    # part c
    '''
    Yes, the key points are roughly the same as SIFT is scale independent. It only looks at the texture of objects.
    '''

    # part d
    detector = get_sift_detector()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = detector.detectAndCompute(img, None)
    kp2, des2 = detector.detectAndCompute(resized, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    good = sorted(good, key=lambda x: x[0].distance)

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img, kp1, resized, kp2, good[:5], img, flags=2)
    cv2.imwrite(name + '_D.jpg', img3)


# task2(img.copy(), 'task2')


def task3(img, name):
    # part a
    rotated = imutils.rotate_bound(img, angle=60)

    # part b
    detector = get_sift_detector()
    kp2, des2 = detector.detectAndCompute(rotated, None)

    rotated_dist = cv2.drawKeypoints(rotated, kp2, rotated)
    cv2.imwrite(name + '_B.jpg', rotated_dist)

    # part c
    '''
    Yes, the key points are roughly the same as SIFT is orientation independent. It only looks at the texture of objects. But we can see little variations. 
    '''

    # part d
    kp1, des1 = detector.detectAndCompute(img, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.70 * n.distance:
            good.append([m])

    good = sorted(good, key=lambda x: x[0].distance)

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img, kp1, rotated, kp2, good[:5], img, flags=2)
    cv2.imwrite(name + '_D.jpg', img3)


task3(img.copy(), 'task3')