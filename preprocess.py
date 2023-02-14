# Preprocessing image - downscaling, grayscale, gaussian blur and threshold

# Sources: https://pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/ (13.01.23)
# https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/ (13.01.23)
# https://learnopencv.com/opencv-threshold-python-cpp/

# import libraries
import cv2


# open and preprocess image
def prep(path):

    # values for threshold value and max threshold
    thresh = 150
    max_value = 220

    # load the image
    img = cv2.imread(path)

    # apply blur
    # img = cv2.GaussianBlur(img, (31, 31), 0)
    # img = cv2.medianBlur(img, ksize=11)  # large kernel

    # apply threshold
    _, img = cv2.threshold(img, thresh, max_value, cv2.THRESH_BINARY)

    # remove noise
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    bg = cv2.morphologyEx(img, cv2.MORPH_DILATE, se)
    out_gray = cv2.divide(img, bg, scale=255)
    out_binary = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1]

    return out_binary, cv2.imread(path)


# load image and apply morphology
def morph(img):
    # load the image
    img = cv2.imread(img)

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blur
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=33, sigmaY=33)

    # divide
    divide = cv2.divide(gray, blur, scale=255)

    # apply threshold
    thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # save result
    cv2.imwrite("hebrew_text_division.jpg", divide)
    cv2.imwrite("hebrew_text_division_threshold.jpg", thresh)
    cv2.imwrite("hebrew_text_division_morph.jpg", morph)

    return morph


# resize an image by scale_percent%
def scale(img, scale_percent):
    # calculate new width and height
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # return resized image
    return img
