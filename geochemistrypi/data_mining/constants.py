import os

# The number of uploading dataset per user is limited to 5.
MAX_UPLOADS_PER_USER = 5

# the directory in which the package(application) is installed
PACKAGEDIR = os.path.dirname(os.path.realpath(__file__))

# the directory where the built-in data set to be processed stays
BUILT_IN_DATASET_PATH = os.path.join(PACKAGEDIR, "data", "dataset")

# current working directory in which the user activates the application
WORKING_PATH = os.getcwd()

# the root directory where all the output stays
OUTPUT_PATH = os.path.join(WORKING_PATH, "geopi_output")

# the directory where the artifact is saved within the MLflow run's artifact directory
MLFLOW_ARTIFACT_DATA_PATH = "data"
MLFLOW_ARTIFACT_IMAGE_STATISTIC_PATH = os.path.join("image", "statistic")
MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH = os.path.join("image", "model_output")
MLFLOW_ARTIFACT_IMAGE_MAP_PATH = os.path.join("image", "map")

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
    "K-Nearest Neighbors",
    "Support Vector Machine",
    "Decision Tree",
    "Random Forest",
    "Extra-Trees",
    "Gradient Boosting",
    "Xgboost",
    "Multi-layer Perceptron",
    "Lasso Regression",
    "Elastic Net",
    "SGD Regression",
]
CLASSIFICATION_MODELS = [
    "Logistic Regression",
    "Support Vector Machine",
    "Decision Tree",
    "Random Forest",
    "Extra-Trees",
    "Xgboost",
    "Multi-layer Perceptron",
    "Gradient Boosting",
    "K-Nearest Neighbors",
    "Stochastic Gradient Descent",
]
CLUSTERING_MODELS = ["KMeans", "DBSCAN"]
DECOMPOSITION_MODELS = ["PCA", "T-SNE", "MDS"]

# Special AutoML models
NON_AUTOML_MODELS = ["Linear Regression", "Polynomial Regression"]
RAY_FLAML = ["Multi-layer Perceptron"]

IMPUTING_STRATEGY = ["Mean Value", "Median Value", "Most Frequent Value", "Constant(Specified Value)"]

FEATURE_SCALING_STRATEGY = ["Min-max Scaling", "Standardization"]

SAMPLE_BALANCE_STRATEGY = ["Over Sampling", "Under Sampling", "Oversampling and Undersampling"]

CUSTOMIZE_LABEL_STRATEGY = ["Automatic Coding", "Custom Numeric Labels", "Custom Non-numeric Labels"]

FEATURE_SELECTION_STRATEGY = ["GenericUnivariateSelect", "SelectKBest"]

CALCULATION_METHOD_OPTION = ["Micro", "Macro", "Weighted"]
