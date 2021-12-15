import os

# current package path
WORKING_PATH = os.getcwd()

# the directory where data set processed is to be stored
DATASET_PATH = os.path.join(WORKING_PATH, "dataset")

# the directory where pictures saved
IMAGE_PATH = os.path.join(WORKING_PATH, "images")

# create the directories if they didn't exist yet
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(IMAGE_PATH, exist_ok=True)

OPTION = ['Yes', 'No']

MODE_OPTION = ['Supervised Learning', 'Unsupervised Learning']

REGRESSION_MODELS = ['Polynomial Regression', 'Xgboost']

IMPUTING_STRATEGY = ['mean', 'median', 'most_frequent']



