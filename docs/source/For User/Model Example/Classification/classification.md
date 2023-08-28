# Classification

Classification is a supervised learning task, in which the training data we feed to the algorithm includes the desired labels. The aim of classification task is to classify each data into the corresponding class. So we have to use dataset with known labels to train a classification model. Then choose one model which has best performance to predict unknown data.

Note：If your task is binary classification, the label must be set to either 0 or 1. All metric values would be calculated from the label 1 by default, such as precision, accurary and so on.

## 1. Preparation
After ensuring the Geochemistry Pi framework has been installed successfully (if not, please see [docs](https://github.com/ZJUEarthData/geochemistrypi/blob/main/docs/source/For%20User/Installation%20Manual.md)), we run the python framework in command line interface to process our program: If you do not input own data, you can run:
```
geochemistrypi data-mining
```
If you prepare to input own data, you can run:
```
geochemistrypi data-mining --data your_own_data_set.xlsx
```
As an example for classification, at the beginning, we should enter 2.
```
-*-*- Built-in Data Option-*-*-
1 - Data For Regression
2 - Data For Classification
3 - Data For Clustering
4 - Data For Dimensional Reduction
(User) ➜ @Number:
```

**Import data**

By entering the number 2, you can see the information below on your screen (if using the data provided by us):
```
Successfully loading the built-in data set 'Data_Classification.xlsx'.
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

**World Map Projection**

After pressing enter to move forward, we can choose whether to project specific elements in the world map.
```
-*-*- World Map Projection -*-*-
World Map Projection for A Specific Element Option:
1 - Yes
2 - No
(Plot) ➜ @Number:
```
If you enter 1, you can choose one specific element column in the dataset to draw automatically
```
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
We enter 10 as an example, after entering the number 10, we can get the path where the image is saved:
```
Save figure 'Map Projection - AL2O3(WT%)' in D:\test\geopi_output\clf\example\artifacts\image\map.
```
![Map Projection - AL2O3(WT%)](https://github.com/Darlx/image/raw/main/img1/Map%20Projection%20-%20AL2O3(WT%25).png)
<font color=gray size=1><center>Figure 1 Map Projection - AL2O3(WT%)</center></font>

After that，you can make a choice again in the following option:
```
Do you want to continue to project a new element in the World Map?
1 - Yes
2 - No
(Plot) ➜ @Number:
```
If we enter 1, we can make another drawing, and if we enter 2, we can step into next mode. We enter 2 here to move forward.

## 2. Data processing
**Data select**

It's not necessary to deal with all the data, so in this part, we can choose the data according to our task. In this example, we choose column 1 as the Y, and column 8-16 as X, so we enter 3; [8, 16] :
```
-*-*- Data Selection -*-*-
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
Format 2: "xx", such as "7" --> you want to deal with the columns 7
@input: 3; [8, 16]
```
After pressing enter, we can see the chosen data:
```
Index - Column Name
3 - Label
8 - SIO2(WT%)
9 - TIO2(WT%)
10 - AL2O3(WT%)
11 - CR2O3(WT%)
12 - FEOT(WT%)
13 - CAO(WT%)
14 - MGO(WT%)
15 - MNO(WT%)
16 - NA2O(WT%)
--------------------
(Press Enter key to move forward.)
```
And we can see the details of the chosen data and some statistic information, in this process, Correlation Plot, Distribution Histogram and Distribution Histogram After Log Transformation are generated and saved under artifacts\image\statistic folder:
```
Basic Statistical Information:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2011 entries, 0 to 2010
Data columns (total 10 columns):
 #   Column      Non-Null Count  Dtype
---  ------      --------------  -----
 0   Label       2011 non-null   int64
 1   SIO2(WT%)   2011 non-null   float64
 2   TIO2(WT%)   2011 non-null   float64
 3   AL2O3(WT%)  2011 non-null   float64
 4   CR2O3(WT%)  2011 non-null   float64
 5   FEOT(WT%)   2011 non-null   float64
 6   CAO(WT%)    2011 non-null   float64
 7   MGO(WT%)    2011 non-null   float64
 8   MNO(WT%)    2011 non-null   float64
 9   NA2O(WT%)   2011 non-null   float64
dtypes: float64(9), int64(1)
memory usage: 157.2 KB
None
Some basic statistic information of the designated data set:
            Label    SIO2(WT%)    TIO2(WT%)   AL2O3(WT%)   CR2O3(WT%)    FEOT(WT%)     CAO(WT%)     MGO(WT%)     MNO(WT%)    NA2O(WT%)
count  2011.00000  2011.000000  2011.000000  2011.000000  2011.000000  2011.000000  2011.000000  2011.000000  2011.000000  2011.000000
mean      0.73446    52.110416     0.411454     4.627858     0.756601     3.215889    21.442025    16.242567     0.091170     0.974539
std       0.44173     2.112777     0.437279     2.268114     0.581543     1.496576     2.325046     2.506461     0.051188     0.632556
min       0.00000     0.218000     0.000000     0.010000     0.000000     1.281000     0.097000     5.500000     0.000000     0.000000
25%       0.00000    51.350271     0.166500     3.531000     0.490500     2.535429    20.532909    15.411244     0.068500     0.363000
50%       1.00000    52.200000     0.320000     4.923000     0.695000     2.920000    21.600000    16.180000     0.089299     0.850000
75%       1.00000    52.980000     0.512043     5.921734     0.912950     3.334500    22.421935    16.893500     0.108000     1.545363
max       1.00000    56.301066     6.970000    48.223000    15.421000    18.270000    26.090000    49.230000     1.090000     5.920000
Successfully calculate the pair-wise correlation coefficient among the selected columns.
Save figure 'Correlation Plot' in D:\test\geopi_output\clf\example\artifacts\image\statistic.
Successfully store 'Correlation Plot' in 'Correlation Plot.xlsx' in D:\test\geopi_output\clf\example\artifacts\image\statistic.
Successfully draw the distribution plot of the selected columns.
Save figure 'Distribution Histogram' in D:\test\geopi_output\clf\example\artifacts\image\statistic.
Successfully store 'Distribution Histogram' in 'Distribution Histogram.xlsx' in D:\test\geopi_output\clf\example\artifacts\image\statistic.
Successfully draw the distribution plot after log transformation of the selected columns.
Save figure 'Distribution Histogram After Log Transformation' in D:\test\geopi_output\clf\example\artifacts\image\statistic.
Successfully store 'Distribution Histogram After Log Transformation' in 'Distribution Histogram After Log Transformation.xlsx' in
D:\test\geopi_output\clf\example\artifacts\image\statistic.
Successfully store 'Data Original' in 'Data Original.xlsx' in D:\test\geopi_output\clf\example\artifacts\data.
```
**Deal with missing value**

After choosing the data, we can use some imputation techniques to deal with the missing value, we can see the values information below:
```
-*-*- Imputation -*-*-
Check which column has null values:
--------------------
Label         False
SIO2(WT%)     False
TIO2(WT%)     False
AL2O3(WT%)    False
CR2O3(WT%)    False
FEOT(WT%)     False
CAO(WT%)      False
MGO(WT%)      False
MNO(WT%)      False
NA2O(WT%)     False
dtype: bool
--------------------
The ratio of the null values in each column:
--------------------
Label         0.0
SIO2(WT%)     0.0
TIO2(WT%)     0.0
AL2O3(WT%)    0.0
CR2O3(WT%)    0.0
FEOT(WT%)     0.0
CAO(WT%)      0.0
MGO(WT%)      0.0
MNO(WT%)      0.0
NA2O(WT%)     0.0
dtype: float64
--------------------
Note: you don't need to deal with the missing values, we'll just pass this step!
(Press Enter key to move forward.)
```
In this example we don't need to deal with the missing value, so just move forward.

**Feature engineering**

Then, you can construct some features with entering 1 here:
```
-*-*- Feature Engineering -*-*-
The Selected Data Set:
--------------------
Index - Column Name
1 - Label
2 - SIO2(WT%)
3 - TIO2(WT%)
4 - AL2O3(WT%)
5 - CR2O3(WT%)
6 - FEOT(WT%)
7 - CAO(WT%)
8 - MGO(WT%)
9 - MNO(WT%)
10 - NA2O(WT%)
--------------------
Feature Engineering Option:
1 - Yes
2 - No
(Data) ➜ @Number:
```
After entering 1, the first thing to do is naming our new feature, in this example, just call it “new feature”. And we also need to build the formula as “b + c + d *e”:
```
-*-*- Feature Engineering -*-*-
The Selected Data Set:
--------------------
Index - Column Name
1 - Label
2 - SIO2(WT%)
3 - TIO2(WT%)
4 - AL2O3(WT%)
5 - CR2O3(WT%)
6 - FEOT(WT%)
7 - CAO(WT%)
8 - MGO(WT%)
9 - MNO(WT%)
10 - NA2O(WT%)
--------------------
Feature Engineering Option:
1 - Yes
2 - No
(Data) ➜ @Number: 1
Selected data set:
a - Label
b - SIO2(WT%)
c - TIO2(WT%)
d - AL2O3(WT%)
e - CR2O3(WT%)
f - FEOT(WT%)
g - CAO(WT%)
h - MGO(WT%)
i - MNO(WT%)
j - NA2O(WT%)
Name the constructed feature (column name), like 'NEW-COMPOUND':
@input: new feature
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
@input: b + c + d *e
(Press Enter key to move forward.)
```
We can see our new feature is presented:
```
Successfully construct a new feature "new feature".
0       54.137300
1       53.167700
2       53.553400
3       53.486800
4       52.500000
          ...
2006    56.443383
2007    59.844304
2008    56.145577
2009    51.140800
2010    50.110000
Name: new feature, Length: 2011, dtype: float64
(Press Enter key to move forward.)
```
And all the features are shown:
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2011 entries, 0 to 2010
Data columns (total 11 columns):
 #   Column       Non-Null Count  Dtype
---  ------       --------------  -----
 0   Label        2011 non-null   int64
 1   SIO2(WT%)    2011 non-null   float64
 2   TIO2(WT%)    2011 non-null   float64
 3   AL2O3(WT%)   2011 non-null   float64
 4   CR2O3(WT%)   2011 non-null   float64
 5   FEOT(WT%)    2011 non-null   float64
 6   CAO(WT%)     2011 non-null   float64
 7   MGO(WT%)     2011 non-null   float64
 8   MNO(WT%)     2011 non-null   float64
 9   NA2O(WT%)    2011 non-null   float64
 10  new feature  2011 non-null   float64
dtypes: float64(10), int64(1)
memory usage: 172.9 KB
None
Some basic statistic information of the designated data set:
            Label    SIO2(WT%)    TIO2(WT%)   AL2O3(WT%)   CR2O3(WT%)    FEOT(WT%)     CAO(WT%)     MGO(WT%)     MNO(WT%)    NA2O(WT%)  new feature
count  2011.00000  2011.000000  2011.000000  2011.000000  2011.000000  2011.000000  2011.000000  2011.000000  2011.000000  2011.000000  2011.000000
mean      0.73446    52.110416     0.411454     4.627858     0.756601     3.215889    21.442025    16.242567     0.091170     0.974539    56.594077
std       0.44173     2.112777     0.437279     2.268114     0.581543     1.496576     2.325046     2.506461     0.051188     0.632556    15.618082
min       0.00000     0.218000     0.000000     0.010000     0.000000     1.281000     0.097000     5.500000     0.000000     0.000000    40.503315
25%       0.00000    51.350271     0.166500     3.531000     0.490500     2.535429    20.532909    15.411244     0.068500     0.363000    54.665847
50%       1.00000    52.200000     0.320000     4.923000     0.695000     2.920000    21.600000    16.180000     0.089299     0.850000    55.963100
75%       1.00000    52.980000     0.512043     5.921734     0.912950     3.334500    22.421935    16.893500     0.108000     1.545363    57.673152
max       1.00000    56.301066     6.970000    48.223000    15.421000    18.270000    26.090000    49.230000     1.090000     5.920000   744.027883
(Press Enter key to move forward.)
```

After constructing a new feature, we can enter 1 to construct another or enter 2 to move forward, and we enter 2 here:
```
Do you want to continue to construct a new feature?
1 - Yes
2 - No
(Data) ➜ @Number:2
```
And all the selected and constructed data will be stored in the path below:
```
Successfully store 'Data Selected Imputed Feature-Engineering' in 'Data Selected Imputed Feature-Engineering.xlsx' in D:\test\geopi_output\clf\example\artifacts\data.
```

## 3. Train-Test Data Preparation

Then we can move forward to next mode, we need to choose the mode here to process our data, in this example, the task is classification, so we enter 2 here.
```
-*-*- Mode Selection -*-*-
1 - Regression
2 - Classification
3 - Clustering
4 - Dimensional Reduction
(Model) ➜ @Number:
```
Before we start the classification model training, we have to specify our X and Y data set. in the example of our selected data set, we take column [2,11] as our X set and column 1 as Y.

**Show you the X data**
```
-*-*- Data Split - X Set and Y Set -*-*-
Divide the processing data set into X (feature value) and Y (target value) respectively.
Selected sub data set to create X data set:
--------------------
Index - Column Name
1 - Label
2 - SIO2(WT%)
3 - TIO2(WT%)
4 - AL2O3(WT%)
5 - CR2O3(WT%)
6 - FEOT(WT%)
7 - CAO(WT%)
8 - MGO(WT%)
9 - MNO(WT%)
10 - NA2O(WT%)
11 - new feature
--------------------
The selected X data set:
Select the data range you want to process.
Input format:
Format 1: "[**, **]; **; [**, **]", such as "[1, 3]; 7; [10, 13]" --> you want to deal with the columns 1, 2, 3, 7, 10, 11, 12, 13
Format 2: "xx", such as "7" --> you want to deal with the columns 7
@input: [2,11]
--------------------
Index - Column Name
2 - SIO2(WT%)
3 - TIO2(WT%)
4 - AL2O3(WT%)
5 - CR2O3(WT%)
6 - FEOT(WT%)
7 - CAO(WT%)
8 - MGO(WT%)
9 - MNO(WT%)
10 - NA2O(WT%)
11 - new feature
--------------------
Successfully create X data set.
The Selected Data Set:
      SIO2(WT%)  TIO2(WT%)  AL2O3(WT%)  CR2O3(WT%)  FEOT(WT%)   CAO(WT%)   MGO(WT%)  MNO(WT%)  NA2O(WT%)  new feature
0     53.640000   0.400000    0.140000    0.695000  11.130000  20.240000  11.290000    0.2200   2.590000    54.137300
1     52.740000   0.386000    0.060000    0.695000  12.140000  20.480000  10.300000    0.5000   2.250000    53.167700
2     51.710000   0.730000    2.930000    0.380000   6.850000  22.420000  13.470000    0.2400   1.200000    53.553400
3     50.870000   0.780000    2.870000    0.640000   7.530000  22.450000  12.860000    0.1900   1.190000    53.486800
4     50.920000   0.710000    2.900000    0.300000   6.930000  22.620000  13.280000    0.2000   1.230000    52.500000
...         ...        ...         ...         ...        ...        ...        ...       ...        ...          ...
2006  52.628866   0.409385    5.612482    0.606707   2.202400  21.172240  15.056981    0.0456   1.753544    56.443383
2007  52.535656   0.422012    5.384972    1.278862   2.093113  21.150105  14.841571    0.0349   1.710571    59.844304
2008  52.163411   0.665545    4.965511    0.667931   2.202465  21.600643  14.999107    0.0723   1.741574    56.145577
2009  44.940000   3.930000    8.110000    0.280000   6.910000  22.520000  12.170000    0.0500   0.670000    51.140800
2010  46.750000   3.360000    6.640000    0.000000   7.550000  22.540000  11.400000    0.1700   0.680000    50.110000

[2011 rows x 10 columns]
Basic Statistical Information:
Some basic statistic information of the designated data set:
         SIO2(WT%)    TIO2(WT%)   AL2O3(WT%)   CR2O3(WT%)    FEOT(WT%)     CAO(WT%)     MGO(WT%)     MNO(WT%)    NA2O(WT%)  new feature
count  2011.000000  2011.000000  2011.000000  2011.000000  2011.000000  2011.000000  2011.000000  2011.000000  2011.000000  2011.000000
mean     52.110416     0.411454     4.627858     0.756601     3.215889    21.442025    16.242567     0.091170     0.974539    56.594077
std       2.112777     0.437279     2.268114     0.581543     1.496576     2.325046     2.506461     0.051188     0.632556    15.618082
min       0.218000     0.000000     0.010000     0.000000     1.281000     0.097000     5.500000     0.000000     0.000000    40.503315
25%      51.350271     0.166500     3.531000     0.490500     2.535429    20.532909    15.411244     0.068500     0.363000    54.665847
50%      52.200000     0.320000     4.923000     0.695000     2.920000    21.600000    16.180000     0.089299     0.850000    55.963100
75%      52.980000     0.512043     5.921734     0.912950     3.334500    22.421935    16.893500     0.108000     1.545363    57.673152
max      56.301066     6.970000    48.223000    15.421000    18.270000    26.090000    49.230000     1.090000     5.920000   744.027883
Successfully store 'X Without Scaling' in 'X Without Scaling.xlsx' in D:\test\geopi_output\clf\example\artifacts\data.
(Press Enter key to move forward.)
```
**Feature Scaling on X data**

We can also do feature scaling here just by entering 1, and two methods can be applied, we select Min-max Scaling here, so just enter 1:
```
-*-*- Feature Scaling on X Set -*-*-
1 - Yes
2 - No
(Data) ➜ @Number:1
Which strategy do you want to apply?
1 - Min-max Scaling
2 - Standardization
(Data) ➜ @Number:1
```
**Show you the scaling X data**
```
Data Set After Scaling:
      SIO2(WT%)  TIO2(WT%)  AL2O3(WT%)  CR2O3(WT%)  FEOT(WT%)  CAO(WT%)  MGO(WT%)  MNO(WT%)  NA2O(WT%)  new feature
0      0.952551   0.057389    0.002696    0.045068   0.579728  0.774939  0.132403  0.201835   0.437500     0.019380
1      0.936504   0.055380    0.001037    0.045068   0.639178  0.784173  0.109764  0.458716   0.380068     0.018001
2      0.918138   0.104735    0.060565    0.024642   0.327800  0.858808  0.182255  0.220183   0.202703     0.018550
3      0.903160   0.111908    0.059320    0.041502   0.367826  0.859962  0.168306  0.174312   0.201014     0.018455
4      0.904052   0.101865    0.059942    0.019454   0.332509  0.866503  0.177910  0.183486   0.207770     0.017052
...         ...        ...         ...         ...        ...       ...       ...       ...        ...          ...
2006   0.934522   0.058735    0.116203    0.039343   0.054235  0.810804  0.218545  0.041835   0.296207     0.022657
2007   0.932860   0.060547    0.111484    0.082930   0.047802  0.809953  0.213619  0.032018   0.288948     0.027492
2008   0.926223   0.095487    0.102784    0.043313   0.054239  0.827286  0.217222  0.066330   0.294185     0.022234
2009   0.797424   0.563845    0.168004    0.018157   0.331332  0.862655  0.152527  0.045872   0.113176     0.015120
2010   0.829698   0.482066    0.137515    0.000000   0.369003  0.863425  0.134919  0.155963   0.114865     0.013655

[2011 rows x 10 columns]
Basic Statistical Information:
Some basic statistic information of the designated data set:
         SIO2(WT%)    TIO2(WT%)   AL2O3(WT%)   CR2O3(WT%)    FEOT(WT%)     CAO(WT%)     MGO(WT%)     MNO(WT%)    NA2O(WT%)  new feature
count  2011.000000  2011.000000  2011.000000  2011.000000  2011.000000  2011.000000  2011.000000  2011.000000  2011.000000  2011.000000
mean      0.925278     0.059032     0.095780     0.049063     0.113891     0.821184     0.245657     0.083642     0.164618     0.022872
std       0.037672     0.062737     0.047044     0.037711     0.088091     0.089449     0.057317     0.046961     0.106851     0.022200
min       0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
25%       0.911724     0.023888     0.073030     0.031807     0.073838     0.786208     0.226646     0.062844     0.061318     0.020131
50%       0.926875     0.045911     0.101902     0.045068     0.096474     0.827261     0.244226     0.081925     0.143581     0.021975
75%       0.940783     0.073464     0.122617     0.059202     0.120872     0.858883     0.260542     0.099083     0.261041     0.024405
max       1.000000     1.000000     1.000000     1.000000     1.000000     1.000000     1.000000     1.000000     1.000000     1.000000
Successfully store 'X With Scaling' in 'X With Scaling.xlsx' in D:\test\geopi_output\clf\example\artifacts\data.
(Press Enter key to move forward.)
```
**Select Y data**

Just enter 1 to choose column 1 as Y:
```
-*-*- Data Split - X Set and Y Set-*-*-
Selected sub data set to create Y data set:
--------------------
Index - Column Name
1 - Label
2 - SIO2(WT%)
3 - TIO2(WT%)
4 - AL2O3(WT%)
5 - CR2O3(WT%)
6 - FEOT(WT%)
7 - CAO(WT%)
8 - MGO(WT%)
9 - MNO(WT%)
10 - NA2O(WT%)
11 - new feature
--------------------
The selected Y data set:
Notice: Normally, please choose only one column to be tag column Y, not multiple columns.
Notice: For classification model training, please choose the label column which has distinctive integers.
Select the data range you want to process.
Input format:
Format 1: "[**, **]; **; [**, **]", such as "[1, 3]; 7; [10, 13]" --> you want to deal with the columns 1, 2, 3, 7, 10, 11, 12, 13
Format 2: "xx", such as "7" --> you want to deal with the columns 7
@input: 1
```
**Show you the Y data**
```
Index - Column Name
1 - Label
--------------------
Successfully create Y data set.
The Selected Data Set:
      Label
0         1
1         1
2         1
3         1
4         1
...     ...
2006      0
2007      0
2008      0
2009      0
2010      0

[2011 rows x 1 columns]
Basic Statistical Information:
Some basic statistic information of the designated data set:
            Label
count  2011.00000
mean      0.73446
std       0.44173
min       0.00000
25%       0.00000
50%       1.00000
75%       1.00000
max       1.00000
Successfully store 'Y' in 'Y.xlsx' in D:\test\geopi_output\clf\example\artifacts\data.
(Press Enter key to move forward.)
```
**Split the data**

Then we have to split our data set in to training data and testing data, we can simply spedcify the spliting ratio in the command line,we set a ratio at 0.2 here, so enter 0.2:
```
-*-*- Data Split - Train Set and Test Set -*-*-
Notice: Normally, set 20% of the dataset aside as test set, such as 0.2
(Data) ➜ @Test Ratio:0.2
```


## 4. Model Selection
After preparing the data, the next is selecting the mode, in current version, 7 classification models are provided, and we can also choose to use them all. In this example, we select Xgboost as our model, so enter 6:
```
-*-*- Model Selection -*-*-:
1 - Logistic Regression
2 - Support Vector Machine
3 - Decision Tree
4 - Random Forest
5 - Extra-Trees
6 - Xgboost
7 - Multi-layer Perceptron
8 - All models above to be trained
Which model do you want to apply?(Enter the Corresponding Number)
(Model) ➜ @Number:6
```
Geochemistrypi integrated the autoML library for its machine learning tasks, so we can simply choose to employ automated machine learning with respect to this algorithm.
```
Do you want to employ automated machine learning with respect to this algorithm?(Enter the Corresponding Number):
1 - Yes
2 - No
(Model) ➜ @Number:1
```
And we can choose whether to implement sample balance on train set, we enter 1 here:
```
-*-*- Sample Balance on Train Set -*-*-
1 - Yes
2 - No
(Data) ➜ @Number:1
```
After implementing sample balance, we need to select the strategy, three strategies are provided, we select over sampling here, so enter 1:
```
Which strategy do you want to apply?
1 - Over Sampling
2 - Under Sampling
3 - Oversampling and Undersampling
(Data) ➜ @Number: 1
```
Then we can see the result:
```
Train Set After Resampling:
      SIO2(WT%)  TIO2(WT%)  AL2O3(WT%)  CR2O3(WT%)  FEOT(WT%)  CAO(WT%)  MGO(WT%)  MNO(WT%)  NA2O(WT%)  new feature  Label
0      0.971523   0.003013    0.023210    0.001491   0.018483  0.931020  0.261628  0.022018   0.129054     0.020252      1
1      0.896741   0.124821    0.148508    0.058362   0.212432  0.724926  0.232106  0.064220   0.211149     0.024633      1
2      0.946974   0.086154    0.117833    0.069187   0.046818  0.816951  0.216236  0.048569   0.345189     0.027712      0
3      0.978566   0.016069    0.036069    0.002075   0.037495  0.930135  0.248136  0.025688   0.110304     0.020985      1
4      0.896206   0.070301    0.143322    0.044744   0.163576  0.764937  0.224102  0.091743   0.244932     0.021664      1
...         ...        ...         ...         ...        ...       ...       ...       ...        ...          ...    ...
2343   0.911541   0.157819    0.098729    0.081058   0.052328  0.879198  0.237823  0.036697   0.042230     0.025442      0
2344   0.912611   0.074605    0.163649    0.053823   0.118842  0.700304  0.244683  0.100917   0.312500     0.025548      0
2345   0.898346   0.088953    0.142700    0.051229   0.097651  0.800331  0.221816  0.082569   0.312500     0.022970      0
2346   0.942778   0.071316    0.111327    0.041986   0.059161  0.828201  0.218321  0.039083   0.294413     0.023549      0
2347   0.923131   0.047346    0.102877    0.051229   0.079993  0.847267  0.206494  0.091743   0.292230     0.022377      0

[2348 rows x 11 columns]
Basic Statistical Information:
Some basic statistic information of the designated data set:
         SIO2(WT%)    TIO2(WT%)   AL2O3(WT%)   CR2O3(WT%)    FEOT(WT%)     CAO(WT%)     MGO(WT%)     MNO(WT%)    NA2O(WT%)  new feature        Label
count  2348.000000  2348.000000  2348.000000  2348.000000  2348.000000  2348.000000  2348.000000  2348.000000  2348.000000  2348.000000  2348.000000
mean      0.925848     0.060676     0.099046     0.049687     0.111062     0.819494     0.241247     0.084753     0.184083     0.023047     0.500000
std       0.034103     0.061700     0.046935     0.035291     0.090621     0.078273     0.048134     0.045013     0.109267     0.020521     0.500107
min       0.000000     0.000000     0.000000     0.000000     0.004297     0.000000     0.048708     0.000000     0.000000     0.000000     0.000000
25%       0.911926     0.030129     0.077137     0.034988     0.068258     0.783454     0.222765     0.064151     0.066216     0.020442     0.000000
50%       0.925750     0.051650     0.106310     0.048635     0.090539     0.822123     0.237534     0.081482     0.197635     0.022400     0.500000
75%       0.939357     0.076578     0.130048     0.058459     0.116973     0.856196     0.256117     0.098165     0.281564     0.024699     1.000000
max       0.999876     1.000000     1.000000     1.000000     0.981164     1.000000     0.999154     0.458716     1.000000     1.000000     1.000000
Successfully store 'X Train After Sample Balance' in 'X Train After Sample Balance.xlsx' in D:\test\geopi_output\clf\example\artifacts\data.
Successfully store 'Y Train After Sample Balance' in 'Y Train After Sample Balance.xlsx' in D:\test\geopi_output\clf\example\artifacts\data.
(Press Enter key to move forward.)
```

## 5. Results

```
*-**-* Xgboost is running ... *-**-*
Expected Functionality:
+  Model Score
+  Confusion Matrix
+  Cross Validation
+  Model Prediction
+  Model Persistence
+  Precision Recall Curve
+  ROC Curve
+  Two-dimensional Decision Boundary Diagram
+  Permutation Importance Diagram
+  Feature Importance Diagram
...
```
The program will implement cross validation during the training with k-Folds=10. Finally, the confusion matrix, feature importance diagram, ROC Curve, Precision Recall Curve and Permutation Importance Diagram will be saved under artifacts/image/model_output folder. The classification report, cross validation report and model score report will be saved under metrics folder. In the meantime, trained Xgboost model will be saved under the artifacts/model folder and hyper parameters will be saved under the parameters folder.

![Confusion_Matrix_Xgboost](https://github.com/Darlx/image/blob/main/Confusion%20Matrix%20-%20Xgboost.png)
<font color=gray size=1><center>Figure 2 Confusion Matrix - Xgboost</center></font>

![ROC_Curve_Xgboost.png](https://github.com/Darlx/image/blob/main/ROC%20Curve%20-%20Xgboost.png)

<font color=gray size=1><center>Figure 3 Classification - Xgboost - ROC Curve</center></font>

![Precision_Recall_Curve_Xgboost.png](https://github.com/Darlx/image/blob/main/Precision%20Recall%20Curve%20-%20Xgboost.png)
<font color=gray size=1><center>Figure 4 Classification - Xgboost - Precision Recall Curve</center></font>

![Permutation_Importance_Xgboost.png](https://github.com/Darlx/image/blob/main/Permutation%20Importance%20-%20Xgboost.png)
<font color=gray size=1><center>Figure 5 Classification - Xgboost - Permutation Importance</center></font>

![Feature_Importance_Map.png](https://github.com/Darlx/image/blob/main/Feature%20Importance%20-%20Xgboost.png)
<font color=gray size=1><center>Figure 6 Classification - Xgboost - Feature Importance</center></font>
