import os
# from process.regress import RegressionModelSelection
# from process.classify import ClassificationModelSelection
# from process.cluster import ClusteringModelSelection
# from process.reduct import DimensionalReductionModelSelection


# current package path
WORKING_PATH = os.getcwd()

# the directory where the data set to be processed stays
DATASET_PATH = os.path.join(WORKING_PATH, "dataset")

# the directory where the data set produced stays
DATASET_OUTPUT_PATH = os.path.join(WORKING_PATH, "output")

# the directory where pictures saved
MODEL_OUTPUT_IMAGE_PATH = os.path.join(WORKING_PATH, "images", "model_output")
STATISTIC_IMAGE_PATH = os.path.join(WORKING_PATH, "images", "statistic")
MAP_IMAGE_PATH = os.path.join(WORKING_PATH, "images", "map")

# create the directories if they didn't exist yet
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(MODEL_OUTPUT_IMAGE_PATH, exist_ok=True)
os.makedirs(STATISTIC_IMAGE_PATH, exist_ok=True)
os.makedirs(DATASET_OUTPUT_PATH, exist_ok=True)
os.makedirs(MAP_IMAGE_PATH, exist_ok=True)

OPTION = ['Yes', 'No']

MODE_OPTION = ['Regression', 'Classification', 'Clustering', 'Dimensional Reduction']

REGRESSION_MODELS = ['Polynomial Regression', 'Xgboost']
CLASSIFICATION_MODELS = ['Polynomial Regression', 'Xgboost']
CLUSTERING_MODELS = ['KMeans']
DIMENSIONAL_REDUCTION_MODELS = ['PCA']

# Modes2Models = {1: REGRESSION_MODELS, 2: CLASSIFICATION_MODELS, 3: CLUSTERING_MODELS, 4: DIMENSIONAL_REDUCTION_MODELS}
# Modes2Initiators = {1: RegressionModelSelection, 2: ClassificationModelSelection, 3: ClusteringModelSelection, 4: DimensionalReductionModelSelection}

IMPUTING_STRATEGY = ['mean', 'median', 'most_frequent']



