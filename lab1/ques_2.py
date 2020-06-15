import matplotlib.pyplot as plt
from ques_1 import show_image, read_image_grey, path, contrast_stretching


def get_list(img):
    l = []
    h = img.shape[0]
    w = img.shape[1]

    for row in range(h):
        for col in range(w):
            l.append(img[row][col])
    return l


if __name__ == '__main__':
    # normal image

    # load image
    img = read_image_grey(path)
    show_image(path, img)

    # plot histogram
    plt.hist(get_list(img), bins=255)
    plt.ylabel('No of times')
    plt.show()

    # transformed image

    # load image
    img = contrast_stretching(img)
    show_image('transformed high contrast image', img)

    # plot histogram
    plt.hist(get_list(img), bins=255)
    plt.ylabel('No of times')
    plt.show()

