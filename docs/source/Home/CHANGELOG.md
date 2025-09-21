# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

+ MLOps core of continuous training in web interface
+ More new algorithms and new processing techniques


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



[ unreleased ]: https://github.com/ZJUEarthData/geochemistrypi
[ 0.7.0 ]: https://github.com/ZJUEarthData/Geochemistrypi/compare/v0.6.1...v0.7.0
[ 0.6.1 ]: https://github.com/ZJUEarthData/geochemistrypi/compare/v0.6.0...v0.6.1
[ 0.6.0 ]: https://github.com/ZJUEarthData/geochemistrypi/compare/v0.5.0...v0.6.0
[ 0.5.0 ]: https://github.com/ZJUEarthData/geochemistrypi/compare/v0.4.0...v0.5.0
[ 0.4.0 ]: https://github.com/ZJUEarthData/geochemistrypi/compare/v0.3.0...v0.4.0
[ 0.3.0 ]: https://github.com/ZJUEarthData/geochemistrypi/compare/v0.2.1...v0.3.0
[ 0.2.1 ]: https://github.com/ZJUEarthData/geochemistrypi/compare/v0.2.0...v0.2.1
[ 0.2.0 ]: https://github.com/ZJUEarthData/geochemistrypi/compare/v0.1.0...v0.2.0
[ 0.1.0 ]: https://github.com/ZJUEarthData/geochemistrypi/releases/tag/v0.1.0
