import os

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
GEO_IMAGE_PATH = os.path.join(WORKING_PATH, "images", "geochemistry")

# the directory where the trained model saved
MODEL_PATH = os.path.join(WORKING_PATH, "trained_models")

# create the directories if they didn't exist yet
# os.makedirs(DATASET_PATH, exist_ok=True)
# os.makedirs(MODEL_OUTPUT_IMAGE_PATH, exist_ok=True)
# os.makedirs(STATISTIC_IMAGE_PATH, exist_ok=True)
# os.makedirs(DATASET_OUTPUT_PATH, exist_ok=True)
# os.makedirs(MAP_IMAGE_PATH, exist_ok=True)
# os.makedirs(GEO_IMAGE_PATH, exist_ok=True)
# os.makedirs(MODEL_PATH, exist_ok=True)

# Tell which section the user is currently in on the UML
SECTION = ['User', 'Data', 'Model', 'Plot']

OPTION = ['Yes', 'No']
DATA_OPTION = ['Own Data', 'Testing Data (Built-in)']
TEST_DATA_OPTION = ['Data For Regression', 'Data For Classification',
                    'Data For Clustering', 'Data For Dimensional Reduction']
MODE_OPTION = ['Regression', 'Classification', 'Clustering', 'Dimensional Reduction']

# The model provided to use
REGRESSION_MODELS = ['Linear Regression', 'Polynomial Regression', 'Support Vector Machine', 'Decision Tree',
                     'Random Forest', 'Extra-Trees', 'Xgboost', 'Deep Neural Networks']
CLASSIFICATION_MODELS = ['Logistic Regression', 'Support Vector Machine', 'Decision Tree', 'Random Forest', 'Xgboost']
CLUSTERING_MODELS = ['KMeans', 'DBSCAN']
DECOMPOSITION_MODELS = ['Principal Component Analysis']
NON_AUTOML_MODELS = ['Linear Regression', 'Polynomial Regression']
RAY_FLAML = ['Deep Neural Networks']

IMPUTING_STRATEGY = ['Mean', 'Median', 'Most Frequent']
