# import the necessary packages
import os

# initialize the base path to the input documents dataset
BASE_PATH = "datasets/denoising-dirty-documents"

# define the path to the training directories
TRAIN_PATH = os.path.sep.join([BASE_PATH, "train"])
CLEANED_PATH = os.path.sep.join([BASE_PATH, "train_cleaned"])

# define the path to our output features CSV file then initialize
# the sampling probability for a given row
FEATURES_PATH = "datasets/denoising-dirty-documents/denoiser_features.csv"
SAMPLE_PROB = 0.02

# define the path to our document denoiser model
MODEL_PATH = "abcd/denoiser/models/denoiser.pickle"