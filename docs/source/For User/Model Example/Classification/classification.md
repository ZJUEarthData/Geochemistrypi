# Classification

Classification is a supervised learning task, in which the training data we feed to the algorithm includes the desired labels. The aim of classification task is to classify each data into the corresponding class. So we have to use dataset with known labels to train a classification model. Then choose one model which has best performance to predict unknown data.

Note：If your task is binary classification, the label must be set to either 0 or 1. All metric values would be calculated from the label 1 by default, such as precision, accurary and so on.

## 1. Train-Test Data Preparation
**Choose the mode you need to use.**

```
-*-*- Mode Options -*-*-
1 - Regression
2 - Classification
3 - Clustering
4 - Dimensional Reduction
(Model) ➜ @Number: 2
(Press Enter key to move forward.)
```

Before we start the classfication model training, we have to specify our X and Y data set. in the example of our selected data set, we take column [2,9] as our X set and column 1 as Y.

**Select X data**
```
The selected X data set:
Select the data range you want to process.
Input format:
Format 1: "[**, **]; **; [**, **]", such as "[1, 3]; 7; [10, 13]" --> you want to deal with the columns 1, 2, 3, 7, 10, 11, 12, 13
Format 2: "xx", such as "7" --> you want to deal with the columns 7
@input: [2,9]
```
**Show you the X data**
```
--------------------
Index - Column Name
2 - AL2O3(WT%)
3 - CR2O3(WT%)
4 - FEOT(WT%)
5 - CAO(WT%)
6 - MGO(WT%)
7 - MNO(WT%)
8 - NA2O(WT%)
9 - new feature
--------------------
Successfully create X data set.
The Selected Data Set:
      AL2O3(WT%)  CR2O3(WT%)  FEOT(WT%)   CAO(WT%)   MGO(WT%)  MNO(WT%)  NA2O(WT%)  new feature
0       0.140000    0.695000  11.130000  20.240000  11.290000    0.2200   2.590000    11.227300
1       0.060000    0.695000  12.140000  20.480000  10.300000    0.5000   2.250000    12.181700
2       2.930000    0.380000   6.850000  22.420000  13.470000    0.2400   1.200000     7.963400
3       2.870000    0.640000   7.530000  22.450000  12.860000    0.1900   1.190000     9.366800
4       2.900000    0.300000   6.930000  22.620000  13.280000    0.2000   1.230000     7.800000
...

Successfully store 'X Without Scaling' in 'X Without Scaling.xlsx' in C:\Users\12396\output\data.
(Press Enter key to move forward.)
```
**Feature Scaling on X data**
```
-*-*- Feature Scaling on X Set -*-*-
1 - Yes
2 - No
(Data) ➜ @Number: 1
Which strategy do you want to apply?
1 - Min-max Scaling
2 - Standardization
(Data) ➜ @Number: 2
```
**Show you the scaling X data**
```
Data Set After Scaling:
      AL2O3(WT%)  CR2O3(WT%)  FEOT(WT%)  CAO(WT%)  MGO(WT%)  MNO(WT%)  NA2O(WT%)  new feature
0      -1.979166   -0.105953   5.289461 -0.517119 -1.976412  2.517460   2.554499     0.230889
1      -2.014446   -0.105953   5.964503 -0.413869 -2.371489  7.988905   2.016863     0.286829
2      -0.748763   -0.647750   2.428888  0.420731 -1.106443  2.908278   0.356517     0.039582
3      -0.775224   -0.200553   2.883372  0.433637 -1.349875  1.931234   0.340704     0.121839
4      -0.761993   -0.785350   2.482357  0.506772 -1.182266  2.126643   0.403955     0.030004
...

[2011 rows x 8 columns]
```
**Select Y data**
```
--------------------
The selected Y data set:
Note: Normally, only one column is allowed to be tag column, not multiple columns.
Select the data range you want to process.
Input format:
Format 1: "[**, **]; **; [**, **]", such as "[1, 3]; 7; [10, 13]" --> you want to deal with the columns 1, 2, 3, 7, 10, 11, 12, 13
Format 2: "xx", such as "7" --> you want to deal with the columns 7
@input: 1
```
**Show you the Y data**
```
Successfully create Y data set.
The Selected Data Set:
      Label
0         1
1         1
2         1
3         1
4         1
...

Successfully store 'y' in 'y.xlsx' in C:\Users\12396\output\data.
(Press Enter key to move forward.)
```

Then we have to split our data set in to training data and testing data, we can simply spedcify the spliting ratio in the command line:

**Split the data**

    -*-*- Data Split - Train Set and Test Set -*-*-
    Note: Normally, set 20% of the dataset aside as test set, such as 0.2
    (Data) ➜ @Test Ratio: 0.2


## 2. Model Selection

Since the traing and testing data are ready now, we can selelt a model to start our classification. Geochemistrypi provide users with 5 classification models. Here we use xgboost as an example.

    -*-*- Model Selection -*-*-:
    1 - Logistic Regression
    2 - Support Vector Machine
    3 - Decision Tree
    4 - Random Forest
    5 - Xgboost
    6 - All models above to be trained
    Which model do you want to apply?(Enter the Corresponding Number)
    (Model) ➜ @Number: 5
    (Press Enter key to move forward.)

Gechemistrypi integrated the autoML library for its machine learning tasks, so we can simply choose to employ automated machine learning with respect to this algorithm.

    Do you want to employ automated machine learning with respect to this algorithm?(Enter the Corresponding Number):
    1 - Yes
    2 - No
    (Model) ➜ @Number: 1
    (Press Enter key to move forward.)



## 3. Results

    *-**-* Xgboost is running ... *-**-*
    Expected Functionality:
    +  Model Score
    +  Confusion Matrix
    +  Cross Validation
    +  Model Prediction
    +  Model Persistence
    +  Feature Importance
    ...

The programm will implement cross validation during the training with k-Folds=10. Finally, the confusion matrix, feature importance map plot, feature weights Histograms will be saved under output/images/modle_output folder. In the meantime, trained xgboost model will be saved under the output/trained_models folder.

![Confusion_Matrix_Xgboost](https://github.com/ZJUEarthData/geochemistrypi/assets/66779478/9a492532-f148-4db7-a8c6-cc234cb01b5b)

<font color=gray size=1><center>Figure 1 Confusion Matrix - Xgboost</center></font>

![Classification_Xgboost_Feature_Weights_Histograms_Plot.png](https://github.com/ZJUEarthData/geochemistrypi/assets/66779478/23471f65-11b3-4eb4-b744-7c2cc7ed33e7)

<font color=gray size=1><center>Figure 2 Classification - Xgboost - Feature Weights Histograms Plot</center></font>

![Classification_Xgboost_Feature_Importance_Map_Plot.png](https://github.com/ZJUEarthData/geochemistrypi/assets/66779478/bd9c17ff-accb-4f34-9c35-3f09315d70cd)

<font color=gray size=1><center>Figure 3 Classification - Xgboost - Feature Importance Map Plot</center></font>
