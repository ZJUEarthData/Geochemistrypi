# Decomposition
## Principal Component Analysis (PCA)


Principal component analysis is usually known as PCA. PCA is an unsupervised learning method, in which the training data we feed to the algorithm does not need the desired labels. The aim of PCA is to reduce the dimension of high-dimensional input data. For example, there are x1, x2, x3, x4 and y, up to four columns of data. However, not all kinds of data are essential for the label y. So the PCA could be useful to abandon less important x for regressing/classifying y and accelerate data analysis.
<br />
<br />Note : This part would show the whole process of PCA, including data-processing and model-running.

## Preparation

First, after ensuring the Geochemistry Pi framework has been installed successfully (if not, please see docs ), we run the python framework in command line interface to process our program:
If you do not input own data, you can run

```bash
geochemistrypi data-mining
```

If you prepare to input own data, you can run

```bash
geochemistrypi data-mining --data your_own_data_set.xlsx
```

The command line interface would show

```bash
-*-*- Built-in Data Option-*-*-
1 - Data For Regression
2 - Data For Classification
3 - Data For Clustering
4 - Data For Dimensional Reduction
(User) ➜ @Number: 4
```

You have to choose ***Data For Dimensional Reduction*** and press 4 on your own keyboard. The command line interface would show

```bash
Successfully loading the built-in data set 'Data_Decomposition.xlsx'.
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
(Press Enter key to move forward.
```

Here, we just need to press any keyboard to continue.

```bash
World Map Projection for A Specific Element Option:
1 - Yes
2 - No
(Plot) ➜ @Number::
```

We can choose map projection if we need a world map projection for a specific element option. Choose yes, we can choose an element to map. Choose no, skip to the next mode. More information of the map projection can be seen in [map projection](https://pyrolite.readthedocs.io/en/main/installation.html). In this tutorial, we skip it and gain output as:

```bash
-*-*- Data Selected -*-*-
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
Select the data range you want to process.
Input format:
Format 1: "[**, **]; **; [**, **]", such as "[1, 3]; 7; [10, 13]" --> you want to deal with the columns 1, 2, 3, 7, 10, 11, 12, 13
Format 2: "xx", such as "7" --> you want to deal with the columns 7)
```

Two options are offered. For PCA, the Format 1 method is more useful in multiple dimensional reduction. As a tutorial, we input ***[10, 15]*** as an example.

**Note: [start_col_num, end_col_num]**

The selected feature information would be given

```bash
The Selected Data Set:
     AL2O3(WT%)  CR2O3(WT%)  FEOT(WT%)   CAO(WT%)   MGO(WT%)  MNO(WT%)
0      3.936000       1.440   3.097000  18.546000  18.478000  0.083000
1      3.040000       0.578   3.200000  20.235000  17.277000  0.150000
2      7.016561         NaN   3.172049  20.092611  15.261175  0.102185
3      3.110977         NaN   2.413834  22.083843  17.349203  0.078300
4      6.971044         NaN   2.995074  20.530008  15.562149  0.096700
..          ...         ...        ...        ...        ...       ...
104    2.740000       0.060   4.520000  23.530000  14.960000  0.060000
105    5.700000       0.690   2.750000  20.120000  16.470000  0.120000
106    0.230000       2.910   2.520000  19.700000  18.000000  0.130000
107    2.580000       0.750   2.300000  22.100000  16.690000  0.050000
108    6.490000       0.800   2.620000  20.560000  14.600000  0.070000

[109 rows x 6 columns]
```
After continuing with any key, basic information of selected data would be shown

```bash
Basic Statistical Information:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 109 entries, 0 to 108
Data columns (total 6 columns):
 #   Column      Non-Null Count  Dtype
---  ------      --------------  -----
 0   AL2O3(WT%)  109 non-null    float64
 1   CR2O3(WT%)  98 non-null     float64
 2   FEOT(WT%)   109 non-null    float64
 3   CAO(WT%)    109 non-null    float64
 4   MGO(WT%)    109 non-null    float64
 5   MNO(WT%)    109 non-null    float64
dtypes: float64(6)
memory usage: 5.2 KB
None
Some basic statistic information of the designated data set:
       AL2O3(WT%)  CR2O3(WT%)   FEOT(WT%)    CAO(WT%)    MGO(WT%)    MNO(WT%)
count  109.000000   98.000000  109.000000  109.000000  109.000000  109.000000
mean     4.554212    0.956426    2.962310   21.115756   16.178044    0.092087
std      1.969756    0.553647    1.133967    1.964380    1.432886    0.054002
min      0.230000    0.000000    1.371100   13.170000   12.170000    0.000000
25%      3.110977    0.662500    2.350000   20.310000   15.300000    0.063075
50%      4.720000    0.925000    2.690000   21.223500   15.920000    0.090000
75%      6.233341    1.243656    3.330000   22.185450   16.816000    0.110000
max      8.110000    3.869550    8.145000   25.362000   23.528382    0.400000
Successfully calculate the pair-wise correlation coefficient among the selected columns.
Save figure 'Correlation Plot' in C:\Users\74086\output\images\statistic.
Successfully draw the distribution plot of the selected columns.
Save figure 'Distribution Histogram' in C:\Users\74086\output\images\statistic.
Successfully draw the distribution plot after log transformation of the selected columns.
Save figure 'Distribution Histogram After Log Transformation' in C:\Users\74086\output\images\statistic.
(Press Enter key to move forward.)
```

## NAN value process


Check the NAN values would be helpful for later analysis. In geochemistrypi frame, this option is finished automatically.

```bash
-*-*- Imputation -*-*-
Check which column has null values:
--------------------
AL2O3(WT%)    False
CR2O3(WT%)     True
FEOT(WT%)     False
CAO(WT%)      False
MGO(WT%)      False
MNO(WT%)      False
dtype: bool
--------------------
The ratio of the null values in each column:
--------------------
CR2O3(WT%)    0.100917
AL2O3(WT%)    0.000000
FEOT(WT%)     0.000000
CAO(WT%)      0.000000
MGO(WT%)      0.000000
MNO(WT%)      0.000000
dtype: float64
--------------------
```

Several strategies are offered for processing the missing values, including:

```bash
-*-*- Strategy for Missing Values -*-*-
1 - Mean Value
2 - Median Value
3 - Most Frequent Value
4 - Constant(Specified Value)
Which strategy do you want to apply?
```

We choose the mean Value in this example and the input data be processed automatically as:

```bash
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
MGO(WT%) 1.0
MNO(WT%) 1.0
Note: 'p-value < 0.05' means imputation method doesn't apply to that column.
The columns which rejects null hypothesis: None
Successfully draw the respective probability plot (origin vs. impute) of the selected columns
Save figure 'Probability Plot' in C:\Users\86188\geopi_output\GeoPi - Rock Classification\Xgboost Algorithm - Test 1\artifacts\image\statistic.
Successfully store 'Probability Plot' in 'Probability Plot.xlsx' in C:\Users\86188\geopi_output\GeoPi - Rock Classification\Xgboost Algorithm - Test 1\artifacts\image\statistic.
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 109 entries, 0 to 108
Data columns (total 6 columns):
 #   Column      Non-Null Count  Dtype
---  ------      --------------  -----
 0   AL2O3(WT%)  109 non-null    float64
 1   CR2O3(WT%)  109 non-null    float64
 2   FEOT(WT%)   109 non-null    float64
 3   CAO(WT%)    109 non-null    float64
 4   MGO(WT%)    109 non-null    float64
 5   MNO(WT%)    109 non-null    float64
dtypes: float64(6)
memory usage: 5.2 KB
None
Some basic statistic information of the designated data set:
       AL2O3(WT%)  CR2O3(WT%)   FEOT(WT%)    CAO(WT%)    MGO(WT%)    MNO(WT%)
count  109.000000  109.000000  109.000000  109.000000  109.000000  109.000000
mean     4.554212    0.956426    2.962310   21.115756   16.178044    0.092087
std      1.969756    0.524695    1.133967    1.964380    1.432886    0.054002
min      0.230000    0.000000    1.371100   13.170000   12.170000    0.000000
25%      3.110977    0.680000    2.350000   20.310000   15.300000    0.063075
50%      4.720000    0.956426    2.690000   21.223500   15.920000    0.090000
75%      6.233341    1.170000    3.330000   22.185450   16.816000    0.110000
max      8.110000    3.869550    8.145000   25.362000   23.528382    0.400000
Successfully store 'Data Selected Imputed' in 'Data Selected Imputed.xlsx' in C:\Users\86188\geopi_output\GeoPi - Rock Classification\Xgboost Algorithm - Test 1\artifacts\data.
(Press Enter key to move forward.)
```

## Feature engineering


The next step is the feature engineering options.
```bash
-*-*- Feature Engineering -*-*-
The Selected Data Set:
--------------------
Index - Column Name
1 - AL2O3(WT%)
2 - CR2O3(WT%)
3 - FEOT(WT%)
4 - CAO(WT%)
5 - MGO(WT%)
6 - MNO(WT%)
--------------------
Feature Engineering Option:
1 - Yes
2 - No
```

Feature engineering options are essential for data analysis. We choose Yes and gain

```bash
-*-*- Feature Engineering -*-*-
The Selected Data Set:
--------------------
Index - Column Name
1 - AL2O3(WT%)
2 - CR2O3(WT%)
3 - FEOT(WT%)
4 - CAO(WT%)
5 - MGO(WT%)
6 - MNO(WT%)
--------------------
Feature Engineering Option:
1 - Yes
2 - No
(Data) ➜ @Number: 1
Selected data set:
a - AL2O3(WT%)
b - CR2O3(WT%)
c - FEOT(WT%)
d - CAO(WT%)
e - MGO(WT%)
f - MNO(WT%)
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
@input:
```

Considering actual need for constructing several new geochemical indexes. We can set up some new indexes. Here, we would set up a new index by *AL2O3/CAO* via keyboard options with *a/d*.

```bash
Do you want to continue to construct a new feature?
1 - Yes
2 - No
(Data) ➜ @Number: 2
Successfully store 'Data Before Splitting' in 'Data Before Splitting.xlsx' in C:\Users\74086\output\data.
Exit Feature Engineering Mode.
```

## PCA


Then we can start PCA by selecting Dimensional Reduction and Principal Component Analysis. The kept component number is a hyper-parameter needs to be decided and here we propose the number is 3. Some PCA information is shown on the window.

```bash
-*-*- Hyper-parameters Specification -*-*-
Decide the component numbers to keep:
(Model) ➜ @Number: 3
*-**-* PCA is running ... *-**-*
Expected Functionality:
+  Model Persistence
+  Principal Components
+  Explained Variance Ratio
+  Compositional Bi-plot
+  Compositional Tri-plot
-----* Principal Components *-----
Every column represents one principal component respectively.
Every row represents how much that row feature contributes to each principal component respectively.
The tabular data looks like in format: 'rows x columns = 'features x principal components'.
                 PC1       PC2       PC3
AL2O3(WT%) -0.742029 -0.439057 -0.085773
CR2O3(WT%) -0.007037  0.082531 -0.213232
FEOT(WT%)  -0.173824  0.219858  0.937257
CAO(WT%)    0.624609 -0.620584  0.200722
MGO(WT%)    0.165265  0.605489 -0.168090
MNO(WT%)   -0.003397  0.011160  0.012315
           -0.040834 -0.014650 -0.005382
-----* Explained Variance Ratio *-----
[0.46679568 0.38306839 0.09102234]
-----* 2 Dimensions Data Selection *-----
The software is going to draw related 2d graphs.
Currently, the data dimension is beyond 2 dimensions.
Please choose 2 dimensions of the data below.
1 - PC1
2 - PC2
3 - PC3
Choose dimension - 1 data:
```

By inputting different component numbers, results of PCA are obtained automatically.

![pca.png](https://github.com/ZJUEarthData/geochemistrypi/assets/66779478/68f31f92-a553-4b1c-b22f-89650614fd98)
<font color=gray size=1><center>Figure 1 PCA Example</center></font>
