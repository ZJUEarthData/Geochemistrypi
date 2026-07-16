# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

+ MLOps core of continuous training in web interface
+ More new algorithms and new processing techniques

---

## [0.9.0] — Trailer

### Added

+ **Chemical Modeling** — [Watch Demo](https://www.bilibili.com/video/BV1JgmcBHEYy/?spm_id_from=333.1387.collection.video_card.click&vd_source=cb3969d68c6d244384e336ba1783ea14)
+ **Network Analysis** — [Read Docs](https://geochemistrypi.readthedocs.io/en/latest/For%20User/Model%20Example/Network_Analysis/Network%20Analysis.html)

---

## [0.8.0] — 2026-07-11

### Added

#### Multi-output Regression
+ Predict multiple continuous target variables simultaneously (multiple X → multiple Y)
+ Integrated with FLAML AutoML for automatic hyperparameter optimization
+ Example: simultaneously predict TI(PPM), V(PPM), CR(PPM), NI(PPM) from major elements

#### Multi-output Classification (Experimental)
+ Predict multiple class labels simultaneously (multiple X → multiple Y)
+ XGBoost native multi-strategy support
+ MultiOutputClassifier wrapper for FLAML AutoML

#### OPTICS Clustering Algorithm
+ Density-based clustering (Ordering Points To Identify the Clustering Structure)
+ No need to pre-specify `eps` parameter
+ Detects clusters of varying densities
+ Suitable for complex geochemical spatial data analysis

#### Automated Clustering Evaluation
+ One-click output of clustering results for K=2 to K=10
+ Automatic Silhouette score calculation for each K value
+ Silhouette score visualization
+ Helps users quickly identify the optimal number of clusters

#### Time Series Analysis Module
+ Independent time series analysis module
+ Customizable bin width and bootstrap iterations
+ Subaerial proportion calculation and visualization
+ Bootstrap resampling for uncertainty quantification
+ Successfully reproduces Liu et al. (2024) results

#### New Built-in Dataset
+ `Data_Time_Series.xlsx` for time series analysis demonstration

#### Documentation
+ Time Series user guide and tutorial
+ Updated README with project statistics badges
+ Added Star History chart to README

### Changed

#### Multi-output Support Improvements
+ Feature selection (`GenericUnivariateSelect` / `SelectKBest`) now automatically skipped for multi-output tasks
+ MLflow parameter logging optimized for multi-output hyperparameters
+ Visualization functions now support multi-output (each target gets separate plots)

#### Dependency Updates
+ Basemap version pinned to `1.3.8` for compatibility

#### Documentation
+ README visual enhancements with badges and structured tables
+ Added `Data_Time_Series.xlsx` to built-in dataset table

### Fixed

#### Multi-output Regression
+ Fixed `fit()` call in `MultiOutputRegressor` (positional arguments `X, y` instead of keyword arguments `X_train, y_train`)
+ Fixed MLflow parameter length limit (500 chars) by splitting long hyperparameters
+ Fixed LightGBM compatibility issues in multi-output tasks

#### Multi-output Classification
+ Fixed `fit()` call in `MultiOutputClassifier` (positional arguments `X, y`)
+ Added XGBoost `multi_strategy` parameter support
+ Fixed label customization for multi-output scenarios

#### General
+ Fixed Python 3.9 type annotation compatibility (`dict | None` → `Optional[dict]`)
+ Fixed `silhouette_score` import in clustering module
+ Resolved application data null value problem (PR #426)
+ Fixed SVM and Decision Tree AutoML runtime issues (PR #429)
+ Resolved `GenericUnivariateSelect` error in feature selection when data contains incomplete values

### Removed

+ Dropped Python 3.8 support (now requires Python 3.9+)

---

### 🎯 Highlights

| Feature | Description |
|---------|-------------|
| **Multi-output Regression** | Predict multiple continuous target variables simultaneously |
| **Multi-output Classification** | Predict multiple class labels simultaneously (experimental) |
| **OPTICS Clustering** | Density-based clustering without a preset `eps` parameter |
| **Automatic Clustering Evaluation** | One-click Silhouette scores for K=2–10 |
| **Time Series Analysis** | Independent module with Bootstrap uncertainty quantification |

### 📌 Compatibility

+ **Python**: 3.9, 3.10
+ **macOS**: ✅ Tested
+ **Windows**: ✅ Tested

---

## [0.7.0] — 2025-01-27

### Added

+ New command option to read dataset and save artifacts on the desktop
  + `geochemistrypi data-mining --desktop`
+ Chatbot driven by LLM AI agent for online docs
+ Row identifier column selection for output data
+ Mean normalization technique in feature scaling section
+ One-click download via [.exe](https://github.com/ZJUEarthData/Geochemistrypi/releases/download/v0.7.0_exe/geochemistrypi_v0.7.0.exe)
+ New models:
  + **Anomaly Detection**
    + Local Outlier Factor
  + **Clustering**
    + Mean Shift
+ Video demos:
  + [v0.7.0 Introduction](https://www.bilibili.com/video/BV1TorTYVEgn/) · [YouTube](https://www.youtube.com/watch?v=6IVaO_gq22A)
  + [Regression Demo](https://www.bilibili.com/video/BV1VormYvEt8/) · [YouTube](https://www.youtube.com/watch?v=eTJ-IV1n4QM)
  + [Classification Demo](https://www.bilibili.com/video/BV1ZDrSYjEBv/) · [YouTube](https://www.youtube.com/watch?v=c_eDI2gVTr0)
  + [Installation Guide via EXE](https://www.bilibili.com/video/BV1YmFPe4ESQ/) · [YouTube](https://www.youtube.com/watch?v=LW5Cngcal9Q)
+ Documentation:
  + *EXE Installation and Operation Guide* (User section)
  + *EXE Packaging Process* (Developer section)

### Changed

+ Documentation:
  + *Add New Model to Framework* (Developer section)
  + *Regression Example* (User section)

---

## [0.6.1] — 2024-07-05

### Added

+ Precision-Recall curve

### Changed

+ Silence dependency downloading noise on first launch

### Fixed

+ Precision-Recall vs. Threshold diagram

---

## [0.6.0] — 2024-06-02

### Added

+ Plotting functions: contour, heatmap, 2D scatter for decomposition
+ Prediction for training set
+ Drop rows with missing values by specific columns
+ Summary folder for all produced artifacts in run output
+ New models:
  + **Regression**
    + Ridge Regression
  + **Clustering**
    + Affinity Propagation
+ New mode:
  + **Anomaly Detection**
    + Isolation Forest
+ Documentation:
  + Mind map of all options in README
  + Citation info
  + Anomaly detection example

### Changed

+ Formula display for linear models (regression & classification) based on target value type
+ Built-in inference data only for regression and classification
+ Documentation:
  + Installation manual
  + Clustering example

### Fixed

+ Invalid YAML file when launching MLflow interface
+ Online docs layout mismatch

---

## [0.5.0] — 2024-01-14

### Added

+ Missing value processing with three options
+ Fixed random state for all models
+ New models:
  + **Regression**
    + Bayesian Ridge Regression
  + **Clustering**
    + Agglomerative Clustering

### Changed

+ Renamed command to implement model inference

---

## [0.4.0] — 2023-12-15

### Added

+ MLOps core of model inference in CLI using transformer pipeline
+ Multi-class and binary label training for all classification models
+ CSV data file import
+ Reduced data storage in decomposition
+ Data selection with null, space, and Chinese parentheses detection
+ Label customization in classification
+ Feature selection
+ Design diagrams of the whole project
+ Feature scaling for unsupervised learning
+ Built-in inference dataset loading
+ Silhouette score frequency diagram for clustering models
+ Two clustering model scores for all clustering models
+ New models:
  + **Regression**
    + Elastic Net
    + Stochastic Gradient Descent
  + **Classification**
    + Gradient Boosting
    + K-Nearest Neighbors
    + Stochastic Gradient Descent

### Changed

+ Lasso Regression with automatic parameter tuning

---

## [0.3.0] — 2023-08-11

### Added

+ Colorful CLI to highlight important information
+ Standardized run-driven experiment workflow
+ Specialized storage mechanism for MLflow-based MLOps lifecycle management
+ Online documentation: Project, User, and Developer sections
+ New models:
  + **Regression**
    + Lasso Regression
    + Gradient Boosting
    + K-Nearest Neighbors
  + **Decomposition**
    + T-SNE
    + MDS
+ Docker deployment configuration
+ Pre-commit CI for code quality

---

## [0.2.1] — 2023-05-01

### Fixed

+ Map projection dependency: replaced `geopandas` with `basemap`

---

## [0.2.0] — 2023-04-19

### Added

+ Manual and automated hyperparameter selection (FLAML + Ray) for all existing models
+ New models:
  + **Classification**
    + Multi-layer Perceptron
    + Extra Trees

---

## [0.1.0] — 2023-02-01

### Added

+ End-to-end automated ML training pipeline with specialized design pattern (MLOps continuous training in CLI)
+ New models:
  + **Regression**
    + Linear Regression
    + Polynomial Regression
    + Decision Tree
    + Extra Trees
    + Random Forest
    + XGBoost
    + Support Vector Machine
    + Multi-layer Perceptron
  + **Classification**
    + Decision Tree
    + Random Forest
    + XGBoost
    + Support Vector Machine
    + Logistic Regression
  + **Clustering**
    + KMeans
    + DBSCAN
  + **Decomposition**
    + Principal Component Analysis
+ GitHub Actions CI for post-commit integration

---

## Version Links

+ [Unreleased](https://github.com/ZJUEarthData/geochemistrypi)
+ [0.8.0](https://github.com/ZJUEarthData/Geochemistrypi/compare/v0.7.0...v0.8.0)
+ [0.7.0](https://github.com/ZJUEarthData/Geochemistrypi/compare/v0.6.1...v0.7.0)
+ [0.6.1](https://github.com/ZJUEarthData/geochemistrypi/compare/v0.6.0...v0.6.1)
+ [0.6.0](https://github.com/ZJUEarthData/geochemistrypi/compare/v0.5.0...v0.6.0)
+ [0.5.0](https://github.com/ZJUEarthData/geochemistrypi/compare/v0.4.0...v0.5.0)
+ [0.4.0](https://github.com/ZJUEarthData/geochemistrypi/compare/v0.3.0...v0.4.0)
+ [0.3.0](https://github.com/ZJUEarthData/geochemistrypi/compare/v0.2.1...v0.3.0)
+ [0.2.1](https://github.com/ZJUEarthData/geochemistrypi/compare/v0.2.0...v0.2.1)
+ [0.2.0](https://github.com/ZJUEarthData/geochemistrypi/compare/v0.1.0...v0.2.0)
+ [0.1.0](https://github.com/ZJUEarthData/geochemistrypi/releases/tag/v0.1.0)
