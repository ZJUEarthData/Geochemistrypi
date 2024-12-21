# -*- coding: utf-8 -*-
from data_mining.cli_pipeline import cli_pipeline
from data_mining.enum import DataSource

"""
Used for internal testing, run in debug mode in IDE to inspect the pipeline
"""

# Mock the scenario where the user uses the built-in dataset for both training and application
#   - Test both continuous training and model inference
# cli_pipeline(training_data_path="", application_data_path="", data_source=DataSource.BUILT_IN)

# Mock the scenario where the user uses the desktop dataset for both training and application
#   - Test both continuous training and model inference
#   - Test continuous training only
cli_pipeline(training_data_path="", application_data_path="", data_source=DataSource.DESKTOP)

# Mock the scenario where the user uses the provided dataset for both training and application
#   - Test both continuous training and model inference
#   - Test continuous training only
# Uncomment the following line to utilize built-in datasets to test the pipeline. Don't forget to modify the path value to be consistent with your own location.
# training_data_path = "/Users/can/Documents/github/work/geo_ml/geochemistrypi/geochemistrypi/data_mining/data/dataset/Data_Classification.xlsx"
# application_data_path = "/Users/can/Documents/github/work/geo_ml/geochemistrypi/geochemistrypi/data_mining/data/dataset/Data_Classification.xlsx"
# cli_pipeline(training_data_path=training_data_path, application_data_path=application_data_path, data_source=DataSource.ANY_PATH)
# cli_pipeline(training_data_path=training_data_path, application_data_path="", data_source=DataSource.ANY_PATH)
