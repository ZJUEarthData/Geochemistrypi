import os

# The number of uploading dataset per user is limited to 5.
MAX_UPLOADS_PER_USER = 5

# current working directory in which the user activates the application
WORKING_PATH = os.getcwd()

# the directory in which the package(application) is installed
PACKAGEDIR = os.path.dirname(os.path.realpath(__file__))

# the directory where the built-in data set to be processed stays
BUILT_IN_DATASET_PATH = os.path.join(PACKAGEDIR, "data", "dataset")

# the root directory where all the output stays
OUTPUT_PATH = os.path.join(WORKING_PATH, "geopi_output")

# the directory where the data set produced stays
DATASET_OUTPUT_PATH = os.path.join(OUTPUT_PATH, "data")

# the directory where pictures saved
MODEL_OUTPUT_IMAGE_PATH = os.path.join(OUTPUT_PATH, "images", "model_output")
STATISTIC_IMAGE_PATH = os.path.join(OUTPUT_PATH, "images", "statistic")
MAP_IMAGE_PATH = os.path.join(OUTPUT_PATH, "images", "map")
GEO_IMAGE_PATH = os.path.join(OUTPUT_PATH, "images", "geochemistry")

# the directory where the trained model saved
MODEL_PATH = os.path.join(OUTPUT_PATH, "trained_models")

# the directory where the data is saved within the MLflow run's artifact directory
MLFLOW_ARTIFACT_DATA_PATH = "data"

# Tell which section the user is currently in on the UML
SECTION = ["User", "Data", "Model", "Plot"]

OPTION = ["Yes", "No"]
DATA_OPTION = ["Own Data", "Testing Data (Built-in)"]
TEST_DATA_OPTION = ["Data For Regression", "Data For Classification", "Data For Clustering", "Data For Dimensional Reduction"]
MODE_OPTION = ["Regression", "Classification", "Clustering", "Dimensional Reduction"]

# The model provided to use
REGRESSION_MODELS = [
    "Linear Regression",
    "Polynomial Regression",
    "Support Vector Machine",
    "Decision Tree",
    "Random Forest",
    "Extra-Trees",
    "Xgboost",
    "Deep Neural Network",
]
CLASSIFICATION_MODELS = [
    "Logistic Regression",
    "Support Vector Machine",
    "Decision Tree",
    "Random Forest",
    "Extra-Trees",
    "Xgboost",
    "Deep Neural Network",
]
CLUSTERING_MODELS = ["KMeans", "DBSCAN"]
DECOMPOSITION_MODELS = ["Principal Component Analysis", "T-SNE"]

# Special AutoML models
NON_AUTOML_MODELS = ["Linear Regression", "Polynomial Regression"]
RAY_FLAML = ["Deep Neural Network"]

IMPUTING_STRATEGY = ["Mean Value", "Median Value", "Most Frequent Value", "Constant(Specified Value)"]

FEATURE_SCALING_STRATEGY = ["Min-max Scaling", "Standardization"]
