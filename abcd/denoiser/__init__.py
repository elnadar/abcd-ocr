# import the necessary packages
from abcd.denoiser.config import denoise_config as config
from abcd.denoiser.denoising import blur_and_threshold
from imutils import paths
import pickle
import random
import cv2
import numpy as np


class Denoiser():

    def __init__(self, image=None, directory=None,
                 samples=0, output=0, save=True):
        if (image == None) and (directory == None):
            assert "Required Variables Error: you should assigne value for `image` variable or `direction`"
        
        self.model = pickle.loads(open(config.MODEL_PATH, "rb").read())
        self.image = image
        self.directory = directory
        self.samples = samples
        self.output = output
        self.save = save
        self.arr = []

    def __loading_image(self, imagePath):
        # load the image, convert it to grayscale, and clone it
        print("[INFO] processing {}".format(imagePath))
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        orig = image.copy()

        # pad the image followed by blurring/thresholding it
        image = cv2.copyMakeBorder(image, 2, 2, 2, 2,
        cv2.BORDER_REPLICATE)
        image = blur_and_threshold(image)

        return orig, image

    def __image_segmanting(self, imagePath):
        # initialize a list to store our ROI features (i.e., 5x5 pixel
        # neighborhoods)
        orig, image = self.__loading_image(imagePath)
        roiFeatures = []

        # slide a 5x5 window across the image
        for y in range(0, image.shape[0]):
            for x in range(0, image.shape[1]):
                # extract the window ROI and grab the spatial dimensions
                roi = image[y:y + 5, x:x + 5]
                (rH, rW) = roi.shape[:2]
                # if the ROI is not 5x5, throw it out
                if rW != 5 or rH != 5:
                    continue
                # our features will be the flattened 5x5=25 pixels from
                # the training ROI
                features = roi.flatten()
                roiFeatures.append(features)

        return orig, roiFeatures

    def __compute(self, imagePath):
        imageName = imagePath.split('/')[-1]
        orig, roiFeatures = self.__image_segmanting(imagePath)
        # use the ROI features to predict the pixels of our new denoised
        # image
        pixels = self.model.predict(roiFeatures)
        # the pixels list is currently a 1D array so we need to reshape
        # it to a 2D array (based on the original input image dimensions)
        # and then scale the pixels from the range [0, 1] to [0, 255]
        pixels = pixels.reshape(orig.shape)
        output = (pixels * 255).astype("uint8")

        if not self.save:
            self.arr.append(output)
        else:
            cv2.imwrite(self.output + imageName, output)

    def __build(self):
        if (self.directory != None):
            # grab the paths to all images in the testing directory and then
            # randomly sample them
            imagePaths = list(paths.list_images(self.directory))

            if self.output == 0:
                self.output = self.directory
            
            if (self.samples != 0):
                random.shuffle(imagePaths)
                imagePaths = imagePaths[:self.samples]

            # loop over the sampled image paths
            for imagePath in imagePaths:
                self.__compute(imagePath)

        else:
            self.__compute(self.image)

    def denoise(self):
        self.__build()
        if not self.save:
            return self.arr
