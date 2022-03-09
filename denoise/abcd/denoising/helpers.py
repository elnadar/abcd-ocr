# import the necessary packages
import numpy as np
import cv2

def blur_and_threshold(image, eps=1e-7):
    """
    blur_and_threshold(image, eps=1e-7)
    
    The blur_and_threshold function accepts two parameters:
        i. image: Our input image that we’ll be pre-processing.
        ii. eps: An epsilon value used to prevent division by zero.
    """
    # apply a median blur to the image and then subtract the blurred
    # image from the original image to approximate the foreground
    blur = cv2.medianBlur(image, 5)
    foreground = image.astype("float") - blur

    # threshold the foreground image by setting any pixels with a
    # value greater than zero to zero
    foreground[foreground > 0] = 0

    # apply min/max scaling to bring the pixel intensities to the
    # range [0, 1]
    min_val = np.min(foreground)
    max_val = np.max(foreground)
    foreground = (foreground - min_val) / (max_val - min_val + eps)

    # return the foreground-approximated image
    return foreground