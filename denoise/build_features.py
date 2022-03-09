# import the necessary packages
from config import denoise_config as config
from abcd.denoising import blur_and_threshold
from imutils import paths
import progressbar
import random
import cv2

# grab the paths to our training images
train_paths = sorted(list(paths.list_images(config.TRAIN_PATH)))
cleaned_paths = sorted(list(paths.list_images(config.CLEANED_PATH)))

# initialize the progress bar
widgets = ["Creating Features: ", progressbar.Percentage(), " ",
    progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(train_paths),
    widgets=widgets).start()

# zip our training paths together, then open the output CSV file for
# writing
imagePaths = zip(train_paths, cleaned_paths)
csv = open(config.FEATURES_PATH, "w")

print()