import os

# The number of uploading dataset per user is limited to 5.
MAX_UPLOADS_PER_USER = 5

# the directory in which the package(application) is installed
PACKAGEDIR = os.path.dirname(os.path.realpath(__file__))

# the directory where the built-in data set to be processed stays
BUILT_IN_DATASET_PATH = os.path.join(PACKAGEDIR, "data", "dataset")

# the directory where the artifact is saved within the MLflow run's artifact directory
MLFLOW_ARTIFACT_DATA_PATH = "data"
MLFLOW_ARTIFACT_IMAGE_STATISTIC_PATH = os.path.join("image", "statistic")
MLFLOW_ARTIFACT_IMAGE_MODEL_OUTPUT_PATH = os.path.join("image", "model_output")
MLFLOW_ARTIFACT_IMAGE_MAP_PATH = os.path.join("image", "map")

# Tell which section the user is currently in on the UML
SECTION = ["User", "Data", "Model", "Plot"]

OPTION = ["Yes", "No"]
DATA_OPTION = ["Own Data", "Testing Data (Built-in)"]
TEST_DATA_OPTION = ["Data For Regression", "Data For Classification", "Data For Clustering", "Data For Dimensional Reduction", "Data For Anomaly Detection"]
MODE_OPTION = ["Regression", "Classification", "Clustering", "Dimensional Reduction", "Anomaly Detection"]
MODE_OPTION_WITH_MISSING_VALUES = ["Regression", "Classification", "Clustering"]

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
    "XGBoost",
    "Multi-layer Perceptron",
    "Lasso Regression",
    "Elastic Net",
    "SGD Regression",
    "BayesianRidge Regression",
    "Ridge Regression",
    # "Bagging Regression",
    # "Decision Tree",
    # Histogram-based Gradient Boosting,
]
CLASSIFICATION_MODELS = [
    "Logistic Regression",
    "Support Vector Machine",
    "Decision Tree",
    "Random Forest",
    "Extra-Trees",
    "XGBoost",
    "Multi-layer Perceptron",
    "Gradient Boosting",
    "K-Nearest Neighbors",
    "Stochastic Gradient Descent",
    # "Bagging Classification",
    # "Decision Tree",
    # Histogram-based Gradient Boosting,
]
CLUSTERING_MODELS = ["KMeans", "DBSCAN", "Agglomerative", "AffinityPropagation", "MeanShift"]
DECOMPOSITION_MODELS = ["PCA", "T-SNE", "MDS"]
ANOMALYDETECTION_MODELS = ["Isolation Forest", "Local Outlier Factor"]

# The model can deal with missing values
# Reference: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values
REGRESSION_MODELS_WITH_MISSING_VALUES = [
    "XGBoost",
    # "Bagging Regression",
    # "Decision Tree",
    # Histogram-based Gradient Boosting,
]
CLASSIFICATION_MODELS_WITH_MISSING_VALUES = [
    "XGBoost",
    # "Bagging Classification",
    # "Decision Tree",
    # Histogram-based Gradient Boosting,
]
CLUSTERING_MODELS_WITH_MISSING_VALUES = [
    # "HDBSCAN"
]


# Special AutoML models
NON_AUTOML_MODELS = ["Linear Regression", "Polynomial Regression"]
RAY_FLAML = ["Multi-layer Perceptron"]

MISSING_VALUE_STRATEGY = ["Drop Rows with Missing Values ", "Impute Missing Values"]

IMPUTING_STRATEGY = ["Mean Value", "Median Value", "Most Frequent Value", "Constant(Specified Value)"]

FEATURE_SCALING_STRATEGY = ["Min-max Scaling", "Standardization", "Mean Normalization"]

SAMPLE_BALANCE_STRATEGY = ["Over Sampling", "Under Sampling", "Oversampling and Undersampling"]

CUSTOMIZE_LABEL_STRATEGY = ["Automatic Coding", "Custom Numeric Labels", "Custom Non-numeric Labels"]

FEATURE_SELECTION_STRATEGY = ["Generic Univariate Select", "Select K Best"]

CALCULATION_METHOD_OPTION = ["Micro", "Macro", "Weighted"]

DROP_MISSING_VALUE_STRATEGY = ["Drop All Rows with Missing Values", "Drop Rows with Missing Values by Specific Columns"]
