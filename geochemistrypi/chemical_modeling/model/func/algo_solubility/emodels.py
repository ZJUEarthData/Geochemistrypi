import numpy as np
import pandas as pd
import xgboost
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


class EModels:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def XGB_params(self):
        params = {
            "colsample_bytree": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "booster": ["gbtree", "dart"],
            "max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10],
            "learning_rate": [0.025, 0.5, 0.1, 0.125, 0.15, 0.2],
            "min_child_weight": [1, 3, 5, 7, 9],
            "n_estimators": [100, 150, 200, 250, 300, 350, 400, 500, 600],
            "base_score": [0.15, 0.25, 0.50, 0.75, 1.0],
        }
        return params

    def RF_params(self):
        params = {
            "n_estimators": [100, 150, 200, 250, 300, 350, 400, 500, 600],
            "max_features": [None, "sqrt"],
            "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9, 10],
            "min_samples_leaf": [1, 2, 4, 6],
            "bootstrap": [True, False],
        }
        return params

    def ET_params(self):
        params = {
            "n_estimators": [100, 150, 200, 250, 300, 350, 400, 500, 600],
            "criterion": ["squared_error", "absolute_error"],
            "max_depth": [2, 8, 16, 32, 50],
            "max_features": [None, "sqrt", "log2"],
            "bootstrap": [True, False],
            "warm_start": [True, False],
        }
        return params

    def ADA_params(self):
        params = {"n_estimators": [100, 400, 700, 1000, 1300, 1600], "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.25, 0.5], "loss": ["linear", "square", "exponential"]}
        return params

    def DT_params(self):
        params = {
            "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
            "splitter": ["best", "random"],
            "max_features": [None, "sqrt", "log2"],
            "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "max_leaf_nodes": [2, 3, 5, 7, 9],
            "min_samples_split": [2, 3, 4, 5, 7],
        }
        return params

    def SVR_params(self):
        params = {
            "kernel": ["rbf", "linear", "poly"],
            "C": [10, 50, 100, 150],
            "gamma": ["scale", "auto"],
            "degree": [2, 3, 4, 5],
            "epsilon": [0.1, 0.2, 0.5, 0.7],
        }
        return params

    def optimize(self):
        xgb = xgboost.XGBRegressor()
        rf = RandomForestRegressor()
        et = ExtraTreesRegressor()
        ada = AdaBoostRegressor()
        dt = DecisionTreeRegressor()
        svr = SVR()

        xgb_opt = RandomizedSearchCV(xgb, param_distributions=self.XGB_params(), cv=5, scoring="r2", n_jobs=-1, n_iter=3, verbose=0, random_state=100)
        rf_opt = RandomizedSearchCV(rf, param_distributions=self.RF_params(), scoring="r2", n_iter=3, cv=5, n_jobs=-1, verbose=0, random_state=101)
        et_opt = RandomizedSearchCV(et, param_distributions=self.ET_params(), cv=5, scoring="r2", n_jobs=-1, n_iter=3, verbose=0, random_state=102)
        ada_opt = RandomizedSearchCV(ada, param_distributions=self.ADA_params(), cv=5, scoring="r2", n_jobs=-1, n_iter=3, verbose=0, random_state=103)
        dt_opt = RandomizedSearchCV(dt, param_distributions=self.DT_params(), cv=5, scoring="r2", n_jobs=-1, n_iter=3, verbose=0, random_state=104)
        svr_opt = RandomizedSearchCV(svr, param_distributions=self.SVR_params(), scoring="r2", cv=5, n_jobs=-1, n_iter=3, verbose=0, random_state=105)
        return xgb_opt, rf_opt, et_opt, ada_opt, dt_opt, svr_opt

    def fit_em(self):
        xgb, rf, et, ada, dt, svr = self.optimize()
        estimators = [xgb, rf, et, ada, dt, svr]
        tuned_models = []
        for estimator in estimators:
            estimator.fit(self.x_train, self.y_train)
            tuned_models.append(estimator.best_estimator_)
        return tuned_models

    def predict_em(self, df):
        tuned_models = self.fit_em()
        y_pred_test = np.empty([len(tuned_models), len(self.x_test)])
        y_pred_df = np.empty([len(tuned_models), len(df)])
        for i, estimator in enumerate(tuned_models):
            y_pred_test[i, :] = estimator.predict(self.x_test)
            y_pred_df[i, :] = estimator.predict(df)
        return y_pred_test, y_pred_df

    def compute_metrics(self, y_true):
        _, y_pred_df = self.predict_em()
        model_names = ["XGB", "RF", "ET", "ADA", "DT", "SVR"]
        r2score, mae, mse, rmse = {}, {}, {}, {}, {}, {}
        for pred, name in zip(y_pred_df, model_names):
            r2score[name] = r2_score(y_true, pred)
            mae[name] = mean_absolute_error(y_true, pred)
            mse[name] = mean_squared_error(y_true, pred)
            rmse[name] = np.sqrt(mean_squared_error(y_true, pred))
        return r2score, mae, mse, rmse

    def fi_em(self):
        tuned_models = self.fit_em()
        fis = [[], [], [], []]
        model_names = ["xgb_fi", "rf_fi", "et_fi", "ada_fi"]
        f_imp = pd.DataFrame(columns=model_names)
        for i, model in enumerate(tuned_models[:-2]):
            fis[i].append(model.feature_importances_)
            f_imp.iloc[:, i] = pd.DataFrame(model.feature_importances_, index=self.train_df.columns)
        return fis, f_imp

    def plot_fi_em(self):
        model_names = ["xgb_fi", "rf_fi", "et_fi", "ada_fi"]
        _, f_imp = self.fi_em()
        f_imp = f_imp.rename_axis("Features").reset_index()
        import matplotlib.pyplot as mpl

        f_imp.plot(x="Features", y=model_names, kind="bar", figsize=(11, 7))
        mpl.ylabel("Model Importances")
        mpl.title("Feature Importance Plots")
        mpl.grid(False)
        mpl.show()

    def plot_results_em(self):
        y_pred = self.predict_em()
        model_names = ["XGB", "RF", "ET", "ADA", "DT", "SVM"]
        import matplotlib.pyplot as mpl

        for pred, models in zip(y_pred, model_names):
            mpl.figure(figsize=(11, 7))
            mpl.scatter(self.y_test, pred)
            mpl.xlabel("True Sulfur test values")
            mpl.ylabel("Predictions on the test set")
            mpl.title("Model fit for {}".format(models))
            mpl.grid(False)
            mpl.show()
        return y_pred
