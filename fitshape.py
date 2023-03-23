import cv2
import imutils
import numpy as np
from imutils import perspective

# display width and height of bounding rectangle
def display(x, y, w, h, image):

    # display width
    cv2.putText(image, "{:.1f}px".format(w), (int(x+100), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 0, 0), 2)

    # display height
    cv2.putText(image, "{:.1f}px".format(h), (int(x), int(y+h)), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 0, 200), 2)
    

# fit line
def line(contours, image, original=[]):

    # loop over the contours individually
    for cnt in contours:

        # ignore small contours
        if cv2.contourArea(cnt) < 20000:
            continue

        # approximate fitted line
        rows, cols = image.shape[:2]
        [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        lefty = int((-x * vy / vx) + y)
        righty = int(((cols - x) * vy / vx) + y)

        # draw fitted line
        if len(original):
            cv2.line(original, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
        else:
            cv2.line(image, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)

    # return vector components
    return vx, vy


# fit minimum area rectangle
def min_rectangle(contours, image, original=[]):
    # loop over the contours individually
    for cnt in contours:

        # ignore small contours
        if cv2.contourArea(cnt) < 5000:
            continue

        # compute the rotated bounding box of the contour
        rect = cv2.minAreaRect(cnt)
        box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
        box = np.intp(box)

        # order the corners of the bounding box
        box = perspective.order_points(box)

        # draw the outline of the bounding box
        if len(original):
            cv2.drawContours(original, [box.astype("int")], -1, (0, 255, 0), 2)
        else:
            cv2.drawContours(image, [box.astype("int")], -1, (0, 255, 0), 2)


# fit rectangle
def rectangle(contours, image, original=[], min_area = 10000):
    # loop over the contours individually
    for cnt in contours:

        # ignore small contours
        if cv2.contourArea(cnt) < min_area:
            continue

        # get corner coordinates (x, y), height (h) and width (w)
        x, y, w, h = cv2.boundingRect(cnt)

        # draw the outline of the bounding box
        if len(original):
            cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #display(x, y, w, h, original)

    # return coordinates, width and height
    return x, y, w, h


# fit ellipse
def ellipse(contours, image, original=[], min_area = 10000):
    # loop over the contours individually
    for cnt in contours:

        # ignore small contours
        if cv2.contourArea(cnt) < min_area:
            continue

        # get corner coordinates (x, y), height (h) and width (w)
        ellipse = cv2.fitEllipse(cnt)

        # draw the outline of the bounding box
        if len(original):
            cv2.ellipse(original, ellipse, (0, 255, 0), 2)
        else:
            cv2.ellipse(image, ellipse, (0, 255, 0), 2)

        #display(x, y, w, h, original)

    # return coordinates, width and height
    return 0, 0, 0, 0
