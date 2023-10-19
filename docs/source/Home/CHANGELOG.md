# Changelog

All notable changes to this project will be documented in this file.



## [Unreleased]

+ MLOps core of model serving in command line interface
+ MLOps core of continuous training in web interface.



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
[ 0.3.0 ]: https://github.com/ZJUEarthData/geochemistrypi/compare/v0.2.1...v0.3.0
[ 0.2.1 ]: https://github.com/ZJUEarthData/geochemistrypi/compare/v0.2.0...v0.2.1
[ 0.2.0 ]: https://github.com/ZJUEarthData/geochemistrypi/compare/v0.1.0...v0.2.0
[ 0.1.0 ]: https://github.com/ZJUEarthData/geochemistrypi/releases/tag/v0.1.0
