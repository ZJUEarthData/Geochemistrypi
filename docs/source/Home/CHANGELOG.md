# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

+ MLOps core of continuous training in web interface
+ More new algorithms and new processing techniques

## [0.9.0] - Trailer

+ [Chemical Modeling function](https://www.bilibili.com/video/BV1JgmcBHEYy/?spm_id_from=333.1387.collection.video_card.click&vd_source=cb3969d68c6d244384e336ba1783ea14)
+ [Network Analysis](https://geochemistrypi.readthedocs.io/en/latest/For%20User/Model%20Example/Network_Analysis/Network%20Analysis.html)

## [0.8.0] - 2026-07-11

### Added

+ **Multi-output Regression**
  + Support for multiple target variables (multiple X → multiple Y)
  + Integrated with FLAML AutoML for automatic hyperparameter optimization
  + Example: simultaneously predict TI(PPM), V(PPM), CR(PPM), NI(PPM) from major elements
  
+ **Multi-output Classification** (Experimental)
  + Support for multiple target labels (multiple X → multiple Y)
  + XGBoost native multi-strategy support
  + MultiOutputClassifier wrapper for FLAML AutoML

+ **OPTICS Clustering Algorithm**
  + Density-based clustering algorithm (Ordering Points To Identify the Clustering Structure)
  + No need to pre-specify eps parameter
  + Detects clusters of varying densities
  + Suitable for complex geochemical spatial data analysis

+ **Automated Clustering Evaluation**
  + One-click output of clustering results for K=2 to K=10
  + Automatic Silhouette score calculation for each K value
  + Silhouette score visualization for each K value
  + Helps users quickly identify optimal number of clusters

+ **Time Series Analysis Module**
  + Independent time series analysis module
  + Customizable bin width and bootstrap iterations
  + Subaerial proportion calculation and visualization
  + Bootstrap resampling for uncertainty quantification
  + Successful reproduction of Liu et al. (2024) results

+ **New Built-in Dataset**
  + `Data_Time_Series.xlsx` for time series analysis demonstration

+ **Documentation**
  + Time Series user guide and tutorial
  + Updated README with project statistics badges
  + Added Star History chart to README

### Changed

+ **Improved Multi-output Support**
  + Feature selection (GenericUnivariateSelect/SelectKBest) now automatically skipped for multi-output tasks
  + MLflow parameter logging optimized for multi-output hyperparameters
  + Visualization functions now support multi-output outputs (each target gets separate plots)

+ **Dependency Updates**
  + Basemap version pinned to 1.3.8 for compatibility

+ **Documentation**
  + README visual enhancements with badges and structured tables
  + Added Data_Time_Series.xlsx to built-in dataset table

### Fixed

+ **Multi-output Regression**
  + Fixed `fit()` call in `MultiOutputRegressor` (positional arguments X, y instead of keyword arguments X_train, y_train)
  + Fixed MLflow parameter length limit (500 chars) by splitting long hyperparameters
  + Fixed LightGBM compatibility issues in multi-output tasks

+ **Multi-output Classification**
  + Fixed `fit()` call in `MultiOutputClassifier` (positional arguments X, y)
  + Added XGBoost `multi_strategy` parameter support
  + Fixed label customization for multi-output scenarios

+ **General**
  + Fixed Python 3.9 type annotation compatibility (`dict | None` → `Optional[dict]`)
  + Fixed `silhouette_score` import in clustering module
  + Resolved application data null value problem (PR #426)
  + Fixed SVM and Decision Tree AutoML runtime issues (PR #429)
  + Resolved GenericUnivariateSelect error in feature selection when data contains incomplete values



## 🎯 Highlights

| Feature | Description |
|------|------|
| **Multi-output Regression** | Predict multiple continuous target variables at the same time |
| **Multi-output Classification** | Predict multiple class labels at the same time (experimental) |
| **OPTICS Clustering** | Density-based clustering without needing a preset eps parameter |
| **Automatic Clustering Evaluation** | One-click output of Silhouette scores for K=2~10 |
| **Time Series Analysis** | Independent module, supports Bootstrap uncertainty quantification |

### 📌 Version Compatibility

- **Python**: 3.8, 3.9, 3.10
- **macOS**: ✅ Tested and works
- **Windows**: ✅ Tested and works

## [0.7.0] - 2025-01-27

### Added

+ New command option to read dataset and save artifacts on the desktop
  + `geochemistrypi data-mining --desktop`
+ Chatbot driven by LLM AI agent for online docs
+ Row identifier column selection for output data
+ Mean normalization technique in feature scaling section
+ Download geopi by one-click via [.exe](https://github.com/ZJUEarthData/Geochemistrypi/releases/download/v0.7.0_exe/geochemistrypi_v0.7.0.exe)
+ New models:
  + Anomaly detection mode
    + Local outlier factor
  + Clustering mode
    + Mean shift
+ Video demo:
  + Geochemistry π v0.7.0 Introduction Video [[Bilibili]](https://www.bilibili.com/video/BV1TorTYVEgn/?vd_source=27944ab3b73a78970c1a52a5dcbb9140) | [[YouTube]](https://www.youtube.com/watch?v=6IVaO_gq22A)
  + Geochemistry π v0.7.0 for Regression Demo [[Bilibili]](https://www.bilibili.com/video/BV1VormYvEt8/?spm_id_from=333.1387.homepage.video_card.click&vd_source=27944ab3b73a78970c1a52a5dcbb9140) | [[YouTube]](https://www.youtube.com/watch?v=eTJ-IV1n4QM)
  + Geochemistry π v0.7.0 for Classification Demo [[Bilibili]](https://www.bilibili.com/video/BV1ZDrSYjEBv/?spm_id_from=333.1387.homepage.video_card.click&vd_source=27944ab3b73a78970c1a52a5dcbb9140) | [[YouTube]](https://www.youtube.com/watch?v=c_eDI2gVTr0)
  + Geochemistry π - Installation Guide via EXE File [[Bilibili]](https://www.bilibili.com/video/BV1YmFPe4ESQ/?spm_id_from=333.337.search-card.all.click) | [[YouTube]](https://www.youtube.com/watch?v=LW5Cngcal9Q)

+ Documentation
  + *EXE Installation and Operation Guide* in user section
  + *EXE Packaging Process* in developer section

### Changed

+ Documentation
  + *Add New Model to Framework* in developer section
  + *Regression Example* in user section


## [0.6.1] - 2024-07-05

### Added

+ Precision-recall curve

### Changed

+ Silence of dependency downloading when first launching

### Fixed

+ Precision-recall vs. threshold diagram


## [0.6.0] - 2024-06-02

### Added

+ Plotting contour function, plotting heatmap function and plot 2d scatter diagram function for decomposition
+ Prediction for the training set
+ Dropping the rows with missing values by specific columns
+ Summary folder to include all produced artifacts in run's output
+ New Models:
  + Regression Models
    + Ridge Regression
  + Clustering Models
    + Affinity Propagation Clustering
+ New Mode:
  + Anomaly Detection
    +  Isolation Forest
+ Docs:
  + Mind map of all options in README
  + Citation info
  + Anomaly detection algorithm example

### Changed

+ Showing formula function for linear models in both regression and classifiction in terms of the number of the target values' type
+ Built-in inferenc data only for regression and classification
+ Docs:
  + Installation manual
  + Clustering algorithm example

### Fixed

+ Invalid YAML file when launching MLflow interface
+ Online docs layout mismatch


## [0.5.0] - 2024-01-14

### Added

+ Missing value process with three options
+ Fixed random state for all models
+ New Models:
  + Regression Models
    + Bayesian Ridge Regression
  + Clustering Models
    + Agglomerative Clustering

### Changed

+ Renamed command to implement model inference


## [0.4.0] - 2023-12-15

### Added

+ MLOps core of model inference in command line interface using transformer pipeline
+ Multi-class label and binary label training for all classification models
+ CSV data file import
+ Reduced data storage in decomposition
+ Data selection function with null, space and Chinese parentheses dection functionality
+ label customization in classification
+ Feature selection function
+ Design diagrams of the whole project
+ Feature scaling for unsupervised learning
+ Built-in inference dataset loading
+ Silhouette score frequency diagram for all clustering model
+ Two clustering model score for all clustering model
+ New Models:
  + Regression Models
    + Elastic Net
    + Stochastic Gradient Regression
  + Classification Models
    + Gradient Boosting
    + K-Nearest Neighbors
    + Stochastic Gradient Descent

### Changed
+ Lasso regression model with automatic parameter tuning functionality


## [0.3.0] - 2023-08-11

### Added

+ Colourful command line interface to highligh importance stuffs.
+ Standardization of run-driven operation for an experiment.
+ Specialized storage mechanism to achieve the MLOps core of machine learning lifecycle management using MLflow
+ Online documentation, including project section, user section, developer section.
+ New Models:
  + Regression Models
    + Lasso Regression
    + Gradient Boosting
    + K-Nearest Neighbors
  + Decomposition Models
    + T-SNE
    + MDS
+ Docker deployment configuration.
+ Continuous intergration (CI) before git commit using pre-commit.



## [0.2.1] - 2023-05-01

### Fixed

+ Fix map projection dependency by replacing geopandas with basemap.



## [0.2.0] - 2023-04-19

### Added

+ Manual hyper parameters selection and automated hyper parameter selection using FLAML and Ray for every existed models
+ New Models:
  +  Classification Models
    + Multi-layer Perceptron
    + Extra Trees



## [0.1.0] - 2023-02-01

### Added

+ End-to-end cutomized automated machine learning training pipeline with specialized design pattern to achieve the MLOps core of continuous training in command line interface.
+ New Models
  + Regression Models
    + Linear Regression
    + Polynomial Regression
    + Decision Tree
    + Extra Trees
    + Random Forest
    + XGBoost
    + Support Vector Machine
    + Multi-layer Perceptron
  + Classification Models
    + Decision Tree
    + Random Forest
    + XGBoost
    + Support Vector Machine
    + Logistic Regression
  + Clustering Models
    + KMeans
    + DBSCAN
  + Decomposition Models
    + Principle Component Analysis
+ Build up continuous integration (CI) after git commit using Git Action



+ [ unreleased ](https://github.com/ZJUEarthData/geochemistrypi)
+ [ 0.8.0 ](https://github.com/ZJUEarthData/Geochemistrypi/compare/v0.7.0...v0.8.0)
+ [ 0.7.0 ](https://github.com/ZJUEarthData/Geochemistrypi/compare/v0.6.1...v0.7.0)
+ [ 0.6.1 ](https://github.com/ZJUEarthData/geochemistrypi/compare/v0.6.0...v0.6.1)
+ [ 0.6.0 ](https://github.com/ZJUEarthData/geochemistrypi/compare/v0.5.0...v0.6.0)
+ [ 0.5.0 ](https://github.com/ZJUEarthData/geochemistrypi/compare/v0.4.0...v0.5.0)
+ [ 0.4.0 ](https://github.com/ZJUEarthData/geochemistrypi/compare/v0.3.0...v0.4.0)
+ [ 0.3.0 ](https://github.com/ZJUEarthData/geochemistrypi/compare/v0.2.1...v0.3.0)
+ [ 0.2.1 ](https://github.com/ZJUEarthData/geochemistrypi/compare/v0.2.0...v0.2.1)
+ [ 0.2.0 ](https://github.com/ZJUEarthData/geochemistrypi/compare/v0.1.0...v0.2.0)
+ [ 0.1.0 ](https://github.com/ZJUEarthData/geochemistrypi/releases/tag/v0.1.0)
