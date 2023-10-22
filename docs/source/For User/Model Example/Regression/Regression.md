# Regression

## 1. Introduction to Regression

Regression is a type of supervised learning in machine learning that aims to find the relationship between a dependent variable and one or more independent variables. It is used to predict continuous numerical values, such as the price of a house or the number of sales for a particular product.

In regression analysis, the independent variables are also called predictors or features, while the dependent variable is the target variable or response variable. The goal of regression is to build a model that can accurately predict the target variable based on the values of the predictors.

There are several types of regression models, including linear regression, polynomial regression, and logistic regression. In linear regression, the relationship between the dependent variable and independent variable(s) is assumed to be linear, which means that the model can be represented by a straight line. In polynomial regression, the relationship is assumed to be non-linear and can be represented by a polynomial function. In logistic regression, the target variable is binary (e.g., 0 or 1) and the model predicts the probability of the target variable being in one of the two classes.


Overall, regression is a powerful tool for predicting numerical values, and is used in a wide range of applications, from finance and economics to healthcare and social sciences.


## 2. Introduction to Regression function of `Geochemistry π`

### 2.1 Enter the sub-menu of Regression

By running this line of command, the following output should show up on your screen:

```python
-*-*- Built-in Data Option-*-*-
1 - Data For Regression
2 - Data For Classification
3 - Data For Clustering
4 - Data For Dimensional Reduction
(User) ➜ @Number:
```

Enter the serial number of the sub-menu you want to choose and press `Enter`. In this doc, we will focus on the usage of Regression function, to do that, enter `1` and press `Enter`.

```python
-*-*- Built-in Data Option-*-*-
1 - Data For Regression
2 - Data For Classification
3 - Data For Clustering
4 - Data For Dimensional Reduction
(User) ➜ @Number: 1
```

### 2.2 Generate a map projection

By doing the steps in 2.2.1, you should see the following output shows up on your screen (if you use the dataset provided by us):

```python
Successfully loading the built-in data set 'Data_Regression.xlsx'.
--------------------
Index - Column Name
1 - CITATION
2 - SAMPLE NAME
3 - Label
4 - Notes
5 - LATITUDE
6 - LONGITUDE
7 - Unnamed: 6
8 - SIO2(WT%)
9 - TIO2(WT%)
10 - AL2O3(WT%)
11 - CR2O3(WT%)
12 - FEOT(WT%)
13 - CAO(WT%)
14 - MGO(WT%)
15 - MNO(WT%)
16 - NA2O(WT%)
17 - Unnamed: 16
18 - SC(PPM)
19 - TI(PPM)
20 - V(PPM)
21 - CR(PPM)
22 - NI(PPM)
23 - RB(PPM)
24 - SR(PPM)
25 - Y(PPM)
26 - ZR(PPM)
27 - NB(PPM)
28 - BA(PPM)
29 - LA(PPM)
30 - CE(PPM)
31 - PR(PPM)
32 - ND(PPM)
33 - SM(PPM)
34 - EU(PPM)
35 - GD(PPM)
36 - TB(PPM)
37 - DY(PPM)
38 - HO(PPM)
39 - ER(PPM)
40 - TM(PPM)
41 - YB(PPM)
42 - LU(PPM)
43 - HF(PPM)
44 - TA(PPM)
45 - PB(PPM)
46 - TH(PPM)
47 - U(PPM)
--------------------
(Press Enter key to move forward.)
```

After pressing `Enter`to move forward, you will see a question pops up enquiring if you need a world map projection for a specific element option:

```python
World Map Projection for A Specific Element Option:
1 - Yes
2 - No
(Plot) ➜ @Number:
```

By choosing “Yes”, you can then choose one element to be projected in the world map; By choosing “No”, you can skip to the next mode. For demonstrating, we choose “Yes” in this case:

```python
-*-*- Distribution in World Map -*-*-
Select one of the elements below to be projected in the World Map:
--------------------
Index - Column Name
1 - CITATION
2 - SAMPLE NAME
3 - Label
4 - Notes
5 - LATITUDE
6 - LONGITUDE
7 - Unnamed: 6
8 - SIO2(WT%)
9 - TIO2(WT%)
10 - AL2O3(WT%)
11 - CR2O3(WT%)
12 - FEOT(WT%)
13 - CAO(WT%)
14 - MGO(WT%)
15 - MNO(WT%)
16 - NA2O(WT%)
17 - Unnamed: 16
18 - SC(PPM)
19 - TI(PPM)
20 - V(PPM)
21 - CR(PPM)
22 - NI(PPM)
23 - RB(PPM)
24 - SR(PPM)
25 - Y(PPM)
26 - ZR(PPM)
27 - NB(PPM)
28 - BA(PPM)
29 - LA(PPM)
30 - CE(PPM)
31 - PR(PPM)
32 - ND(PPM)
33 - SM(PPM)
34 - EU(PPM)
35 - GD(PPM)
36 - TB(PPM)
37 - DY(PPM)
38 - HO(PPM)
39 - ER(PPM)
40 - TM(PPM)
41 - YB(PPM)
42 - LU(PPM)
43 - HF(PPM)
44 - TA(PPM)
45 - PB(PPM)
46 - TH(PPM)
47 - U(PPM)
--------------------
(Plot) ➜ @Number:
```

Here, we choose “10 - AL2O3(WT%)” as an example, after this, the path to save the image will be presented:

```python
Save figure 'Map Projection - AL2O3(WT%)' in /home/yucheng/output/images/ma
```

![Map Projection - AL2O3(WT%)](https://github.com/ZJUEarthData/geochemistrypi/assets/66779478/0edf28b3-3006-49e6-a2b4-8ddcc7f94306)
<font color=gray size=1><center>Map Projection - AL2O3(WT%)</center></font>

When you see the following instruction:

```python
Do you want to continue to project a new element in the World Map?
1 - Yes
2 - No
(Plot) ➜ @Number:
```

You can choose “Yes” to map another element or choose “No” to exit map mode.

### 2.3 Enter the range of data and check the output

After quitting the map projection mode, you will see the input format from command prompt:

```python
Select the data range you want to process.
Input format:
Format 1: "[**, **]; **; [**, **]", such as "[1, 3]; 7; [10, 13]" --> you want to deal with the columns 1, 2, 3, 7, 10, 11, 12, 13
Format 2: "xx", such as "7" --> you want to deal with the columns 7
@input:
```

Here, we use “[10, 13]” as an example. The values of the elements we choose would be shown on the screen.

```python
--------------------
Index - Column Name
10 - AL2O3(WT%)
11 - CR2O3(WT%)
12 - FEOT(WT%)
13 - CAO(WT%)
--------------------
```

Some basic statistical information of the dataset would also be calculated.

```python
The Selected Data Set:
     AL2O3(WT%)  CR2O3(WT%)  FEOT(WT%)   CAO(WT%)
0      3.936000       1.440   3.097000  18.546000
1      3.040000       0.578   3.200000  20.235000
2      7.016561         NaN   3.172049  20.092611
3      3.110977         NaN   2.413834  22.083843
4      6.971044         NaN   2.995074  20.530008
..          ...         ...        ...        ...
104    2.740000       0.060   4.520000  23.530000
105    5.700000       0.690   2.750000  20.120000
106    0.230000       2.910   2.520000  19.700000
107    2.580000       0.750   2.300000  22.100000
108    6.490000       0.800   2.620000  20.560000

[109 rows x 4 columns]
(Press Enter key to move forward.)
```

```python
Basic Statistical Information:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 109 entries, 0 to 108
Data columns (total 4 columns):
 #   Column      Non-Null Count  Dtype
---  ------      --------------  -----
 0   AL2O3(WT%)  109 non-null    float64
 1   CR2O3(WT%)  98 non-null     float64
 2   FEOT(WT%)   109 non-null    float64
 3   CAO(WT%)    109 non-null    float64
dtypes: float64(4)
memory usage: 3.5 KB
None
Some basic statistic information of the designated data set:
       AL2O3(WT%)  CR2O3(WT%)   FEOT(WT%)    CAO(WT%)
count  109.000000   98.000000  109.000000  109.000000
mean     4.554212    0.956426    2.962310   21.115756
std      1.969756    0.553647    1.133967    1.964380
min      0.230000    0.000000    1.371100   13.170000
25%      3.110977    0.662500    2.350000   20.310000
50%      4.720000    0.925000    2.690000   21.223500
75%      6.233341    1.243656    3.330000   22.185450
max      8.110000    3.869550    8.145000   25.362000
Successfully calculate the pair-wise correlation coefficient among the selected columns.
Save figure 'Correlation Plot' in C:\Users\86188\geopi_output\GeoPi - Rock Classification\Xgboost Algorithm - Test 1\artifacts\image\statistic.
Successfully store 'Correlation Plot' in 'Correlation Plot.xlsx' in C:\Users\86188\geopi_output\GeoPi - Rock Classification\Xgboost Algorithm - Test 1\artifacts\image\statistic.
Successfully draw the distribution plot of the selected columns.
Save figure 'Distribution Histogram' in C:\Users\86188\geopi_output\GeoPi - Rock Classification\Xgboost Algorithm - Test 1\artifacts\image\statistic.
Successfully store 'Distribution Histogram' in 'Distribution Histogram.xlsx' in C:\Users\86188\geopi_output\GeoPi - Rock Classification\Xgboost Algorithm - Test 1\artifacts\image\statistic.
Successfully draw the distribution plot after log transformation of the selected columns.
Save figure 'Distribution Histogram After Log Transformation' in C:\Users\86188\geopi_output\GeoPi - Rock Classification\Xgboost Algorithm - Test 1\artifacts\image\statistic.
Successfully store 'Distribution Histogram After Log Transformation' in 'Distribution Histogram After Log Transformation.xlsx' in C:\Users\86188\geopi_output\GeoPi - Rock Classification\Xgboost Algorithm - Test
1\artifacts\image\statistic.
Successfully store 'Data Original' in 'Data Original.xlsx' in C:\Users\86188\geopi_output\GeoPi - Rock Classification\Xgboost Algorithm - Test 1\artifacts\data.
Successfully store 'Data Selected' in 'Data Selected.xlsx' in C:\Users\86188\geopi_output\GeoPi - Rock Classification\Xgboost Algorithm - Test 1\artifacts\data.
(Press Enter key to move forward.)
```

The function calculates the pairwise correlation coefficients among these elements and create a distribution plot for each element. Here are the plots generated by our example:

![Correlation Plot](https://github.com/ZJUEarthData/geochemistrypi/assets/66779478/be72b8da-aaca-4420-9d78-e8575d6ed8b4)
<font color=gray size=1><center>Correlation Plot</center></font>

![Distribution Histogram](https://github.com/ZJUEarthData/geochemistrypi/assets/66779478/96079dfe-8194-4412-af13-fe44fa1a3dd0)
<font color=gray size=1><center>Distribution Histogram</center></font>

![Distribution Histogram After Log Transformation](https://github.com/ZJUEarthData/geochemistrypi/assets/66779478/2156dc68-2989-44af-aa0b-f47e0ed56012)
<font color=gray size=1><center>Distribution Histogram After Log Transformation</center></font>


### 2.4 Use imputation techniques to deal with the missing values

Below is a brief summary of missing values, you may choose the proper imputation technique to deal with them:

```python
-*-*- Imputation -*-*-
Check which column has null values:
--------------------
AL2O3(WT%)    False
CR2O3(WT%)     True
FEOT(WT%)     False
CAO(WT%)      False
dtype: bool
--------------------
The ratio of the null values in each column:
--------------------
CR2O3(WT%)    0.100917
AL2O3(WT%)    0.000000
FEOT(WT%)     0.000000
CAO(WT%)      0.000000
dtype: float64
--------------------
Note: you'd better use imputation techniques to deal with the missing values.
(Press Enter key to move forward.)
```

Here, we choose ”1 - Mean Values” as our strategy:

```python
-*-*- Strategy for Missing Values -*-*-
1 - Mean Value
2 - Median Value
3 - Most Frequent Value
4 - Constant(Specified Value)
Which strategy do you want to apply?
(Data) ➜ @Number: 1
Successfully fill the missing values with the mean value of each feature column respectively.
(Press Enter key to move forward.)
```

Here, the pragram is performing a hypothesis testing on the imputation method used to fill missing values in a dataset. The null hypothesis is that the distribution of the data set before and after imputing remains the same. The Kruskal Test is used to test this hypothesis, with a significance level of 0.05. Monte Carlo simulation is used with 100 iterations, each with a sample size of half the dataset (54 in this case). The p-values are calculated for each column and the columns that reject the null hypothesis are identified.

```python
-*-*- Hypothesis Testing on Imputation Method -*-*-
Null Hypothesis: The distributions of the data set before and after imputing remain the same.
Thoughts: Check which column rejects null hypothesis.
Statistics Test Method: kruskal Test
Significance Level:  0.05
The number of iterations of Monte Carlo simulation:  100
The size of the sample for each iteration (half of the whole data set):  54
Average p-value:
AL2O3(WT%) 1.0
CR2O3(WT%) 0.9327453056346102
FEOT(WT%) 1.0
CAO(WT%) 1.0
Note: 'p-value < 0.05' means imputation method doesn't apply to that column.
The columns which rejects null hypothesis: None
Successfully draw the respective probability plot (origin vs. impute) of the selected columns
Save figure 'Probability Plot' in C:\Users\86188\geopi_output\GeoPi - Rock Classification\Xgboost Algorithm - Test 1\artifacts\image\statistic.
Successfully store 'Probability Plot' in 'Probability Plot.xlsx' in C:\Users\86188\geopi_output\GeoPi - Rock Classification\Xgboost Algorithm - Test 1\artifacts\image\statistic.
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 109 entries, 0 to 108
Data columns (total 4 columns):
 #   Column      Non-Null Count  Dtype
---  ------      --------------  -----
 0   AL2O3(WT%)  109 non-null    float64
 1   CR2O3(WT%)  109 non-null    float64
 2   FEOT(WT%)   109 non-null    float64
 3   CAO(WT%)    109 non-null    float64
dtypes: float64(4)
memory usage: 3.5 KB
None
Some basic statistic information of the designated data set:
       AL2O3(WT%)  CR2O3(WT%)   FEOT(WT%)    CAO(WT%)
count  109.000000  109.000000  109.000000  109.000000
mean     4.554212    0.956426    2.962310   21.115756
std      1.969756    0.524695    1.133967    1.964380
min      0.230000    0.000000    1.371100   13.170000
25%      3.110977    0.680000    2.350000   20.310000
50%      4.720000    0.956426    2.690000   21.223500
75%      6.233341    1.170000    3.330000   22.185450
max      8.110000    3.869550    8.145000   25.362000
Successfully store 'Data Selected Imputed' in 'Data Selected Imputed.xlsx' in C:\Users\86188\geopi_output\GeoPi - Rock Classification\Xgboost Algorithm - Test 1\artifacts\data.
(Press Enter key to move forward.)
```

A probability plot of the selected columns is also drawn and saved in a specified location.

![Probability Plot](https://github.com/ZJUEarthData/geochemistrypi/assets/66779478/8cab64ac-593a-4f46-bf58-475a59e993e8)

<font color=gray size=1><center>Probability Plot</center></font>

### 2.5 Feature Engineering

Next, you can choose “Yes” for feature engineering option to construct a new feature and select a dataset in the former we choose, or No to exit Feature Engineering mode:

```python
-*-*- Feature Engineering -*-*-
The Selected Data Set:
--------------------
Index - Column Name
1 - AL2O3(WT%)
2 - CR2O3(WT%)
3 - FEOT(WT%)
4 - CAO(WT%)
--------------------
Feature Engineering Option:
1 - Yes
2 - No
(Data) ➜ @Number:
```

After enter “1”, we now is ready to name the constructed feature and build the formula. In this example, we use “newFeature” as the name and we build the formula with “b*c+d”:

```python
Selected data set:
a - AL2O3(WT%)
b - CR2O3(WT%)
c - FEOT(WT%)
d - CAO(WT%)
Name the constructed feature (column name), like 'NEW-COMPOUND':
@input: new Feature
Build up new feature with the combination of basic arithmatic operators, including '+', '-', '*', '/', '()'.
Input example 1: a * b - c
--> Step 1: Multiply a column with b column;
--> Step 2: Subtract c from the result of Step 1;
Input example 2: (d + 5 * f) / g
--> Step 1: Multiply 5 with f;
--> Step 2: Plus d column with the result of Step 1;
--> Step 3: Divide the result of Step 1 by g;
Input example 3: pow(a, b) + c * d
--> Step 1: Raise the base a to the power of the exponent b;
--> Step 2: Multiply the value of c by the value of d;
--> Step 3: Add the result of Step 1 to the result of Step 2;
Input example 4: log(a)/b - c
--> Step 1: Take the logarithm of the value a;
--> Step 2: Divide the result of Step 1 by the value of b;
--> Step 3: Subtract the value of c from the result of Step 2;
You can use mean(x) to calculate the average value.
@input: b*c+d
```

The output is as below:

```python
Successfully construct a new feature "new Feature".
0      23.005680
1      22.084600
2      23.126441
3      24.392497
4      23.394575
         ...
104    23.801200
105    22.017500
106    27.033200
107    23.825000
108    22.656000
Name: new Feature, Length: 109, dtype: float64
(Press Enter key to move forward.)
-----------------
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 109 entries, 0 to 108
Data columns (total 5 columns):
 #   Column       Non-Null Count  Dtype
---  ------       --------------  -----
 0   AL2O3(WT%)   109 non-null    float64
 1   CR2O3(WT%)   109 non-null    float64
 2   FEOT(WT%)    109 non-null    float64
 3   CAO(WT%)     109 non-null    float64
 4   new Feature  109 non-null    float64
dtypes: float64(5)
memory usage: 4.4 KB
None
Some basic statistic information of the designated data set:
       AL2O3(WT%)  CR2O3(WT%)   FEOT(WT%)    CAO(WT%)  new Feature
count  109.000000  109.000000  109.000000  109.000000   109.000000
mean     4.554212    0.956426    2.962310   21.115756    23.853732
std      1.969756    0.524695    1.133967    1.964380     1.596076
min      0.230000    0.000000    1.371100   13.170000    18.474000
25%      3.110977    0.680000    2.350000   20.310000    22.909000
50%      4.720000    0.956426    2.690000   21.223500    23.904360
75%      6.233341    1.170000    3.330000   22.185450    24.763500
max      8.110000    3.869550    8.145000   25.362000    29.231800
(Press Enter key to move forward.)
```

After building the new feature, we can choose the mode to process data, in this doc, we choose “1 - Regression”:

```python
-*-*- Mode Selection -*-*-
1 - Regression
2 - Classification
3 - Clustering
4 - Dimensional Reduction
(Model) ➜ @Number: 1
```

## 3. Regression Model-Running

After entering the Regression menu, we are going to input X Set and Y Set separately, note that the new feature we just created is also in the list:

```python
-*-*- Data Split - X Set and Y Set-*-*-
Divide the processing data set into X (feature value) and Y (target value) respectively.
Selected sub data set to create X data set:
--------------------
Index - Column Name
1 - AL2O3(WT%)
2 - CR2O3(WT%)
3 - FEOT(WT%)
4 - CAO(WT%)
5 - new Feature
--------------------
The selected X data set:
Select the data range you want to process.
Input format:
Format 1: "[**, **]; **; [**, **]", such as "[1, 3]; 7; [10, 13]" --> you want to deal with the columns 1, 2, 3, 7, 10, 11, 12, 13
Format 2: "xx", such as "7" --> you want to deal with the columns 7
@input:
```

After entering the X Set, the prompt of successful operation and basic statistical information would be shown:

```python
uccessfully create X data set.
The Selected Data Set:
     AL2O3(WT%)
0      3.936000
1      3.040000
2      7.016561
3      3.110977
4      6.971044
..          ...
104    2.740000
105    5.700000
106    0.230000
107    2.580000
108    6.490000

[109 rows x 1 columns]
Basic Statistical Information:
Some basic statistic information of the designated data set:
       AL2O3(WT%)
count  109.000000
mean     4.554212
std      1.969756
min      0.230000
25%      3.110977
50%      4.720000
75%      6.233341
max      8.110000
Successfully store 'X Without Scaling' in 'X Without Scaling.xlsx' in /home/yucheng/output/data.
(Press Enter key to move forward.)
```

After this, you may choose to process feature scaling on X Set or not:

```python
-*-*- Feature Scaling on X Set -*-*-
1 - Yes
2 - No
(Data) ➜ @Number:
```

In the similar manner, we then set Y Set and check the related information generated onto the screen.

The next step is to split the data into a training set and a test set. The test set will be used to evaluate the performance of the machine learning model that will be trained on the training set. In this example, we set 20% of the data to be set aside as the test set. This means that 80% of the data will be used as the training set. The data split is important to prevent overfitting of the model on the training data and to ensure that the model's performance can be generalized to new, unseen data:

```python
-*-*- Data Split - Train Set and Test Set -*-*-
Note: Normally, set 20% of the dataset aside as test set, such as 0.2
(Data) ➜ @Test Ratio: 0.2
```

After checking the output, you should be able to see a menu to choose a machine learning model for your data, in this example, we are going to use “2 - Polynomial Regression”:

```python
-*-*- Model Selection -*-*-:
1 - Linear Regression
2 - Polynomial Regression
3 - K-Nearest Neighbors
4 - Support Vector Machine
5 - Decision Tree
6 - Random Forest
7 - Extra-Trees
8 - Gradient Boosting
9 - Xgboost
10 - Multi-layer Perceptron
11 - Lasso Regression
12 - All models above to be trained
Which model do you want to apply?(Enter the Corresponding Number)
(Model) ➜ @Number: 2
```

After choosing the model, the command line may prompt you to provide more specific options in terms of the model you choose, after offering the options, the program is good to go! And you may check the output like this after processing:



```python
*-**-* Polynomial Regression is running ... *-**-*
Expected Functionality:
+  Model Score
+  Cross Validation
+  Model Prediction
+  Model Persistence
+  Predicted vs. Actual Diagram
+  Residuals Diagram
+  Permutation Importance Diagram
+  Polynomial Regression Formula
-----* Model Score *-----
Root Mean Square Error: 1.2981800081993564
Mean Absolute Error: 0.8666537321359384
R2 Score: -0.5692041761356125
Explained Variance Score: -0.5635060495257759
-----* Cross Validation *-----
K-Folds: 10
* Fit Time *
Scores: [0.00217414 0.00214863 0.00225115 0.00212574 0.00201654 0.00203323
 0.00196433 0.00200295 0.00195527 0.00195432]
Mean: 0.0020626306533813475
Standard deviation: 9.940905756158756e-05
-------------
* Score Time *
Scores: [0.00440168 0.00398946 0.00407624 0.0041182  0.00420284 0.00452423
 0.00406241 0.00427079 0.00406742 0.00404215]
Mean: 0.004175543785095215
Standard deviation: 0.0001651057611709732
-------------
* Root Mean Square Error *
Scores: [1.15785222 1.29457522 2.71100276 3.38856833 0.94791697 1.0329962
 1.54759602 1.8725529  1.82623562 0.84039699]
Mean: 1.6619693228088945
Standard deviation: 0.7833005136355865
-------------
* Mean Absolute Error *
Scores: [0.86020769 0.85255076 1.71707909 2.17595274 0.73042456 0.8864327
 1.2754413  1.32740744 1.48587525 0.67660019]
Mean: 1.1987971734378662
Standard deviation: 0.4639441214337496
-------------
* R2 Score *
Scores: [ 0.3821429  -0.12200627 -0.58303497 -0.98544835  0.3240076   0.02309755
 -0.93382518 -9.20857756 -1.11023532 -0.50902637]
Mean: -1.2722905973773913
Standard deviation: 2.6935459556340082
-------------
* Explained Variance Score *
Scores: [ 0.42490745 -0.01768215 -0.54672932 -0.90106814  0.32644583  0.18391296
 -0.92481771 -7.4016756  -0.39601889  0.24420376]
Mean: -0.9008521815781642
Standard deviation: 2.2175052662305945
-------------
-----* Predicted Value Evaluation *-----
Save figure 'Predicted Value Evaluation - Polynomial Regression' in /home/yucheng/output/images/model_output.
-----* True Value vs. Predicted Value *-----
Save figure 'True Value vs. Predicted Value - Polynomial Regression' in /home/yucheng/output/images/model_output.
-----* Polynomial Regression Formula *-----
y = 1.168AL2O3(WT%)+4.677CR2O3(WT%)-0.085AL2O3(WT%)^2-2.572AL2O3(WT%) CR2O3(WT%)-2.229CR2O3(WT%)^2+0.002AL2O3(WT%)^3+0.14AL2O3(WT%)^2 CR2O3(WT%)+0.762AL2O3(WT%) CR2O3(WT%)^2+0.232CR2O3(WT%)^3+1.4708950432993957
-----* Model Prediction *-----
    FEOT(WT%)   CAO(WT%)
0    6.234901  21.516655
1    3.081208  20.471231
2    3.082333  19.539309
3    2.838430  20.666521
4    2.434649  21.558533
5    2.478282  21.784115
6    2.689378  20.075947
7    2.744644  21.954583
8    3.336340  22.054664
9    3.033059  20.288637
10   3.268753  21.438835
11   3.129242  22.290128
12   2.451531  21.640214
13   2.984390  19.752188
14   2.513781  21.035197
15   2.699384  20.676107
16   2.641574  21.844654
17   3.449548  20.632201
18   3.134386  22.138135
19   2.986511  21.673300
20   2.899159  19.943711
21   2.606604  22.146161
Successfully store 'Y Test Predict' in 'Y Test Predict.xlsx' in /home/yucheng/output/data.
-----* Model Persistence *-----
Successfully store the trained model 'Polynomial Regression' in 'Polynomial_Regression_2023-02-24.pkl' in /home/yucheng/output/trained_models.
Successfully store the trained model 'Polynomial Regression' in 'Polynomial_Regression_2023-02-24.joblib' in /home/yucheng/output/trained_models.
```

```
