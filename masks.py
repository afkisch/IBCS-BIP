# import libraries
import cv2
import numpy as np


# apply background mask
def background_mask(image):

    # values for brightness, contrast, threshold value and max threshold
    brightness = 40 #60
    contrast = 100 #80
    thresh = 212 #210
    max_value = 190 #255

    # adjust brightness and contrast
    image = np.int16(image)
    image = image * (contrast / 127 + 1) - contrast + brightness
    image = np.clip(image, 0, 255)
    image = np.uint8(image)

    # apply blur
    #image = cv2.GaussianBlur(image, (31, 31), 0)
    #image = cv2.medianBlur(image, ksize=11)  # large kernel

    # create and apply background mask
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(img_gray, thresh, max_value, cv2.THRESH_BINARY)
    # mask_inv = cv2.bitwise_not(mask)
    img_fg = cv2.bitwise_and(img_gray, img_gray, mask=mask)

    # return original image with the mask applied
    return mask


def eye_mask(image):

    # values for brightness and contrast
    brightness = 60
    contrast = 80

    # adjust brightness and contrast
    image = np.int16(image)
    image = image * (contrast / 127 + 1) - contrast + brightness
    image = np.clip(image, 0, 255)
    image = np.uint8(image)

    # create mask for the eyes
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = cv2.inRange(img_gray, 1, 40)

    # return original image with the mask applied
    return mask
