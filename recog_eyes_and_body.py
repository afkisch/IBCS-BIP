# import libraries
import os
import cv2
import imutils
from imutils import contours
from masks import background_mask
from masks import eye_mask
import numpy as np
from preprocess import prep
from preprocess import scale
from fitshape import display
from fitshape import line
from fitshape import min_rectangle
from fitshape import rectangle

# number of images to be processed
max_index = 2

# name of source folder
src_path = 'medaka'

for i in range(max_index+1):
    
    # load preprocessed and original image
    _ , img = prep('src/' + src_path + '_' + str(i) + '.jpg')
    _ , org = prep('src/' + src_path + '_' + str(i) + '.jpg')

    # apply background mask
    img = background_mask(img)

    # find contours
    cnts, _ = cv2.findContours(img, 1, 2)
    cnts, _ = contours.sort_contours(cnts)

    # fit line in order to determine main axis and angle
    vx, vy = line(cnts, img)

    # calculate angle
    angle = float(180 * np.arctan(vy / vx) / np.pi)

    # rotate image
    rotated = imutils.rotate(org, angle=angle)
    # print('rotate by:', angle)

    # save rotated image
    cv2.imwrite(os.path.join('img', 'rotated_' + str(i) + '.tif'), rotated)

    ##### Detect eyes

    # open rotated image
    _ , img = prep('img/rotated_' + str(i) + '.tif')
    _ , org = prep('img/rotated_' + str(i) + '.tif')

    # apply eye mask
    img = eye_mask(img)

    # find contours
    cnts, _ = cv2.findContours(img, 1, 2)
    cnts, _ = contours.sort_contours(cnts)

    # fit rectangle
    x, y, w, h = rectangle(cnts, img, org)

    print('Dimensions of picture ' + str(i) + ':' + str(w) + 'x' + str(h) + 'px')

    # display width and height of rectangle
    #display(x, y, w, h, org)

    ##### Detect body

    # open rotated image
    _ , img = prep('img/rotated_' + str(i) + '.tif')

    # apply background mask
    img = background_mask(img)

    # perform edge detection
    #img = cv2.Canny(img, 20, 90) #2080

    # perform dilation and erosion to close gaps
    #img = cv2.dilate(img, None, iterations=2) #2
    #img = cv2.erode(img, None, iterations=1)

    cv2.imshow('img', scale(img, 30))

    # find contours
    cnts, _ = cv2.findContours(img, 1, 2)
    cnts, _ = contours.sort_contours(cnts)

    # fit rectangle
    x, y, w, h = rectangle(cnts, img, org, 60000)

    #print('Dimensions of picture ' + str(i) + ':' + str(w) + 'x' + str(h) + 'px')

    # display width and height of rectangle
    #display(x, y, w, h, org)

    # show image
    cv2.imshow('Body and eyes', scale(org, 30))
    cv2.waitKey()

