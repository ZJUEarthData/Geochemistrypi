# Regression

## 1. Introduction to Regression

Regression is a type of supervised learning in machine learning that aims to find the relationship between a dependent variable and one or more independent variables. It is used to predict continuous numerical values, such as the price of a house or the number of sales for a particular product.

In regression analysis, the independent variables are also called predictors or features, while the dependent variable is the target variable or response variable. The goal of regression is to build a model that can accurately predict the target variable based on the values of the predictors.

There are several types of regression models, including linear regression, polynomial regression, and logistic regression. In linear regression, the relationship between the dependent variable and independent variable(s) is assumed to be linear, which means that the model can be represented by a straight line. In polynomial regression, the relationship is assumed to be non-linear and can be represented by a polynomial function. In logistic regression, the target variable is binary (e.g., 0 or 1) and the model predicts the probability of the target variable being in one of the two classes.


Overall, regression is a powerful tool for predicting numerical values, and is used in a wide range of applications, from finance and economics to healthcare and social sciences.


## 2. Introduction to Regression function of Geochemistry π

### 2.1 Enter the sub-menu of Regression

By running this line of command, the following output should show up on your screen:

```python
-*-*- Built-in Data Option-*-*-
1 - Data For Regression
2 - Data For Classification
3 - Data For Clustering
4 - Data For Dimensional Reduction
(User) ➜ @Number: 1
```

Enter the serial number of the sub-menu you want to choose and press `Enter`. In this doc, we will focus on the usage of Regression function, to do that, enter `1` and press `Enter`.


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
(Plot) ➜ @Number:  1
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
(Plot) ➜ @Number: 10
```

Here, we choose “10 - AL2O3(WT%)” as an example, after this, the path to save the image will be presented:

```python
Save figure 'Map Projection - AL2O3(WT%)' in C:\Users\YSQ\geopi_output\Regression\test\artifacts\image\map.
Successfully store 'Map Projection - AL2O3(WT%)'in 'Map Projection - AL2O3(WT%).xlsx' inC:\Users\YSQ\geopi_output\Regression\test\artifacts\image\map.
```

![Map Projection - AL2O3(WT%)](https://github.com/ZJUEarthData/geochemistrypi/assets/162782014/5b642790-99a6-4421-9422-7fd482a6d425)
<font color=gray size=1><center>Map Projection - AL2O3(WT%)</center></font>

When you see the following instruction:

```python
Do you want to continue to project a new element in the World Map?
1 - Yes
2 - No
(Plot) ➜ @Number: 2
```

You can choose “Yes” to map another element or choose “No” to exit map mode. Here, we choose 2 to skip this step.

### 2.3 Enter the range of data and check the output

After quitting the map projection mode, you will see the input format from command prompt:

```python
Select the data range you want to process.
Input format:
Format 1: "[**, **]; **; [**, **]", such as "[1, 3]; 7; [10, 13]" --> you want to deal with the columns 1, 2, 3, 7, 10, 11, 12, 13
Format 2: "xx", such as "7" --> you want to deal with the columns 7
@input: [10,13]
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
Successfully calculate the pair-wise correlation coefficient among the selected columns. Save figure 'Correlation Plot' in C:\Users\YSQ\geopi_output\Regression\test\artifacts\image\statistic.
Successfully store 'Correlation Plot' in 'Correlation Plot.xlsx' in C:\Users\YSQ\geopi_output\Regression\test\artifacts\image\statistic.
...
Successfully store 'Data Original' in 'DataOriginal.xlsx' in C:\Users\YSQ\geopi_output\Regression\test\artifacts\data.
Successfully store 'Data Selected' in 'DataSelected.xlsx' in C:\Users\YSQ\geopi_output\Regression\test\artifacts\data.
```

The function calculates the pairwise correlation coefficients among these elements and create a distribution plot for each element. Here are the plots generated by our example:


![Correlation Plot](https://github.com/ZJUEarthData/geochemistrypi/assets/162782014/5774e386-c1ab-4347-8be0-592e00ab004f)
<font color=gray size=1><center>Correlation Plot</center></font>

![Distribution Histogram](https://github.com/ZJUEarthData/geochemistrypi/assets/162782014/cfdd5c8b-2428-493d-98be-712885a1cde8)
<font color=gray size=1><center>Distribution Histogram</center></font>


![Distribution Histogram After Log Transformation](https://github.com/ZJUEarthData/geochemistrypi/assets/162782014/7ebd82fa-1fb9-4cfe-9b59-9a479a59ca2b)
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

```python
-*-*- Missing Values Process -*-*-
Do you want to deal with the missing values?
1 - Yes
2 - No
(Data) ➜ @Number: 1
```
Here, let's choose 1 to deal with the missing values.

```python
-*-*- Strategy for Missing Values -*-*-
1 - Drop Rows with Missing Values
2 - Impute Missing Values
Notice: Drop the rows with missing values may lead to a significant loss of data if too many
features are chosen.
Which strategy do you want to apply?
(Data) ➜ @Number:1
```
We'll just skip the lines with missing info to keep things simple.

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
(Data) ➜ @Number: 1
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
```
```python
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

This step, we enter b*c+d. And the output is as below:

```python
Successfully construct a new feature new Feature.
0     23.00568
1     22.08460
2     25.43000
3     23.39590
4     22.90900
        ...
93    23.80120
94    22.01750
95    27.03320
96    23.82500
97    22.65600
Name: new Feature, Length: 98, dtype: float64
(Press Enter key to move forward.)
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 98 entries, 0 to 97
Data columns (total 5 columns):
 #   Column       Non-Null Count  Dtype
---  ------       --------------  -----
 0   AL2O3(WT%)   98 non-null     float64
 1   CR2O3(WT%)   98 non-null     float64
 2   FEOT(WT%)    98 non-null     float64
 3   CAO(WT%)     98 non-null     float64
 4   new Feature  98 non-null     float64
dtypes: float64(5)
memory usage: 4.0 KB
None
Some basic statistic information of the designated data set:
       AL2O3(WT%)  CR2O3(WT%)  FEOT(WT%)   CAO(WT%)  new Feature
count   98.000000   98.000000  98.000000  98.000000    98.000000
mean     4.444082    0.956426   2.929757  21.187116    23.883266
std      1.996912    0.553647   1.072481   1.891933     1.644173
min      0.230000    0.000000   1.371100  13.170000    18.474000
25%      3.051456    0.662500   2.347046  20.310000    22.872800
50%      4.621250    0.925000   2.650000  21.310000    23.907180
75%      6.222500    1.243656   3.346500  22.284019    24.795747
max      8.110000    3.869550   8.145000  25.362000    29.231800
(Press Enter key to move forward.)
```

If you feel it's enough, just select 'no' to proceed to the next step. Here, we choose 2.
```python
Do you want to continue to build a new feature?
1 - Yes
2 - No
(Data) ➜ @Number:2
```

```
Successfully store 'Data Selected Dropped-Imputed Feature-Engineering' in 'Data Selected
Dropped-Imputed Feature-Engineering.xlsx' in
C:\Users\YSQ\geopi_output\Regression\test\artifacts\data.
Exit Feature Engineering Mode.
(Press Enter key to move forward.)
```

After building the feature, we can choose the mode to process data, in this doc, we choose “1 - Regression”:

```python
-*-*- Mode Selection -*-*-
1 - Regression
2 - Classification
3 - Clustering
4 - Dimensional Reduction
(Model) ➜ @Number: 1
```

## 3. Model Selection

After entering the Regression menu, we are going to input X Set and Y Set separately, note that the new feature we just created is also in the list:

```python
-*-*- Data Segmentation - X Set and Y Set -*-*-
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
@input: 1
```

After entering the X Set, the prompt of successful operation and basic statistical information would be shown:

```python
Successfully create X data set.
The Selected Data Set:
    AL2O3(WT%)
0        3.936
1        3.040
2        4.220
3        6.980
4        6.250
..         ...
93       2.740
94       5.700
95       0.230
96       2.580
97       6.490

[98 rows x 1 columns]
Basic Statistical Information:
Some basic statistic information of the designated data set:
       AL2O3(WT%)
count   98.000000
mean     4.444082
std      1.996912
min      0.230000
25%      3.051456
50%      4.621250
75%      6.222500
max      8.110000
Successfully store 'X Without Scaling' in 'X Without Scaling.xlsx' in
C:\Users\YSQ\geopi_output\Regression\test\artifacts\data.
(Press Enter key to move forward.)
```
Then, input Y Set like "2 - CR203(WT%)".
```python
-*-*- Data Segmentation - X Set and Y Set-*-*-
Selected sub data set to create Y data set:
--------------------
Index - Column Name
1 - AL2O3(WT%)
2 - CR2O3(WT%)
3 - FEOT(WT%)
4 - CAO(WT%)
5 - new Feature
--------------------
The selected Y data set:
Notice: Normally, please choose only one column to be tag column Y, not multiple columns.
Notice: For classification model training, please choose the label column which has
distinctive integers.
Select the data range you want to process.
Input format:
Format 1: "[**, **]; **; [**, **]", such as "[1, 3]; 7; [10, 13]" --> you want to deal with the columns 1, 2, 3, 7, 10, 11, 12, 13
Format 2: "xx", such as "7" --> you want to deal with the columns 7
@input:2
```

The prompt of successful operation and basic statistical information would be shown:

```python
Successfully create Y data set.
The Selected Data Set:
    CR2O3(WT%)
0        1.440
1        0.578
2        1.000
3        0.830
4        0.740
..         ...
93       0.060
94       0.690
95       2.910
96       0.750
97       0.800

[98 rows x 1 columns]
Basic Statistical Information:
Some basic statistic information of the designated data set:
       CR2O3(WT%)
count   98.000000
mean     0.956426
std      0.553647
min      0.000000
25%      0.662500
50%      0.925000
75%      1.243656
max      3.869550
Successfully store 'Y' in 'Y.xlsx' in
C:\Users\YSQ\geopi_output\Regression\test\artifacts\data.
(Press Enter key to move forward.)
```


After this, you may choose to process feature scaling on X Set or not:

```python
-*-*- Feature Scaling on X Set -*-*-
1 - Yes
2 - No
(Data) ➜ @Number: 2
```

In this doc, we choose 2, and for the next step of feature selection, we also choose option 2.

```python
-*-*- Feature Selection on X set -*-*-
1 - Yes
2 - No
(Data) ➜ @Number:2
```


The next step is to split the data into a training set and a test set. The test set will be used to evaluate the performance of the machine learning model that will be trained on the training set. In this example, we set 20% of the data to be set aside as the test set. This means that 80% of the data will be used as the training set. The data split is important to prevent overfitting of the model on the training data and to ensure that the model's performance can be generalized to new, unseen data:

```python
-*-*- Data Split - Train Set and Test Set -*-*-
Note: Normally, set 20% of the dataset aside as test set, such as 0.2
(Data) ➜ @Test Ratio: 0.2
```

After checking the output, you should be able to see a menu to choose a machine learning model for your data, in this example, we are going to use “7 - Extra-Trees”.

```python
-*-*- Model Selection -*-*-
1 - Linear Regression
2 - Polynomial Regression
3 - K-Nearest Neighbors
4 - Support Vector Machine
5 - Decision Tree
6 - Random Forest
7 - Extra-Trees
8 - Gradient Boosting
9 - XGBoost
10 - Multi-layer Perceptron
11 - Lasso Regression
12 - Elastic Net
13 - SGD Regression
14 - BayesianRidge Regression
15 - All models above to be trained
Which model do you want to apply?(Enter the Corresponding Number)
(Model) ➜ @Number:7
```

We have already set up an automated learning program. You can simply choose option '1' to easily access it.

```python
Do you want to employ automated machine learning with respect
to this algorithm?(Enter the Corresponding Number):
1 - Yes
2 - No
(Model) ➜ @Number:1
```


```python
-*-*- Feature Engineering on Application Data -*-*-
The same feature engineering operation will be applied to the
inference data.
Successfully construct a new feature new Feature.
0           NaN
1           NaN
2           NaN
3     25.430000
4     22.909000
5     23.211800
...
49    25.158800
50    23.342814
51    21.512000
52    25.668000
53    23.801200
54    23.825000
Name: new Feature, dtype: float64
Successfully store 'Application Data Original' in 'Application Data Original.xlsx' in C:\Users\YSQ\geopi_output\GeoPi - Rock Classification\Regression\artifacts\data.
Successfully store 'Application Data Feature-Engineering' in 'Application Data Feature-Engineering.xlsx' in C:\Users\YSQ\geopi_output\GeoPi - Rock Classification\Regression\artifacts\data.
Successfully store 'Application Data Feature-Engineering Selected' in 'Application Data Feature-Engineering Selected.xlsx' in C:\Users\YSQ\geopi_output\GeoPi - Rock Classification\Regression\artifacts\data.
(Press Enter key to move forward.)
```


After moving to the next step, the Extra-Trees algorithm training will run automatically.
This includes functionalities such as Model Scoring, Cross Validation, Predicted vs. Actual Diagram, Residuals Diagram, Permutation Importance Diagram, Feature Importance Diagram, Single Tree Diagram, Model Prediction, and Model Persistence.
You can find the output stored in the specified path.

```python
-*-*- Transform Pipeline Construction -*-*-
Build the transform pipeline according to the previous operations.
Successfully store 'Transform Pipeline Configuration' in 'Transform Pipeline Configuration.txt' in C:\Users\YSQ\geopi_output\GeoPi - Rock Classification\Regression\artifacts.
(Press Enter key to move forward.)

-*-*- Model Inference -*-*-
Use the trained model to make predictions on the application data.
Successfully store 'Application Data Predicted' in 'Application Data Predicted.xlsx' in C:\Users\YSQ\geopi_output\GeoPi - Rock Classification\Regression\artifacts\data.
Successfully store 'Application Data Feature-Engineering Selected Dropped-Imputed' in 'Application Data Feature-Engineering Selected Dropped-Imputed.xlsx' in C:\Users\YSQ\geopi_output\GeoPi - Rock Classification\Regression\artifacts\data.
(Press Enter key to move forward.)
```


![Feature Importance - Extra-Trees](https://github.com/user-attachments/assets/1d3f6177-8495-445f-ae6b-7bee8ab002a7)
<div style="color: gray; font-size: 1em; text-align: center;">Feature Importance - Extra-Trees</div>

![Permutation Importance - Extra-Trees](https://github.com/user-attachments/assets/22d825b4-c58d-4ead-ad6a-35cf7179a426)
<div style="color: gray; font-size: 1em; text-align: center;">Permutation Importance - Extra-Trees</div>

![Predicted vs  Actual Diagram - Extra-Trees](https://github.com/user-attachments/assets/a1adc20d-cfc0-459d-8a05-83c8b877af93)
<div style="color: gray; font-size: 1em; text-align: center;">Predicted vs  Actual Diagram - Extra-Trees</div>

![Residuals Diagram - Extra-Trees](https://github.com/user-attachments/assets/08cf261b-a60f-4bf5-94b1-c7f03e9c3358)
<div style="color: gray; font-size: 1em; text-align: center;">Residuals Diagram - Extra-Trees</div>

![Tree Diagram - Extra-Trees](https://github.com/user-attachments/assets/619d9a3d-9b60-4560-bdfd-3623ba293bff)
<div style="color: gray; font-size: 1em; text-align: center;">Tree Diagram - Extra-Trees</div>
