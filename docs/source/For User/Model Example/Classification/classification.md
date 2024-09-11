# Classification



Classification is a supervised learning task, in which the training data we feed to the algorithm includes the desired labels. The aim of classification task is to classify each data into the corresponding class. So we have to use dataset with known labels to train a classification model. Then choose one model which has best performance to predict unknown data.



Note：If your task is binary classification, the label must be set to either 0 or 1. All metric values would be calculated from the label 1 by default, such as precision, accurary and so on.



## Table of Contents



- [1. Train-Test Data Preparation](#1.Train-Test-Data-Preparation)
- [2. Missing Value Processing](#2.Missing-Value-Processing)
- [3. Data Processing](#3.Data-Processing)
- [4. Model Selection](#4.Model-Selection)



## 1. Train-Test Data Preparation



After [installing](https://geochemistrypi.readthedocs.io/en/latest/For%20User/Installation%20Manual.html) it, the first step is to run the Geochemistry Pi framework in your terminal application.



In this section, we take the built-in dataset as an example by running:



```bash

geochemistrypi data-mining

```



Alternatively, it is perfectly fine if you would like to use your own dataset like:



```bash

geochemistrypi data-mining --data your_own_data_set.xlsx

```



You can choose the appropriate option based on the program's prompts and press the Enter key to select the default option (inside the parentheses).



```bash

Welcome to Geochemistry π!

Initializing...

No Training Data File Provided!

Built-in Data Loading.

No Application Data File Provided!

Built-in Application Data Loading.

✨ Press Ctrl + C to exit our software at any time.

✨ Input Template [Option1/Option2] (Default Value): Input Value

✨ Use Previous Experiment [y/n] (n):

✨ New Experiment (GeoPi - Rock Classification):

  'GeoPi - Rock Classification' is activated.

✨ Run Name (XGBoost Algorithm - Test 1):

(Press Enter key to move forward.)

```



After pressing the Enter key, the program propts the following options to let you ***\*choose the Built-in Training Data\****:



```

-*-*- Built-in Training Data Option-*-*-

1 - Data For Regression

2 - Data For Classification

3 - Data For Clustering

4 - Data For Dimensional Reduction

(User) ➜ @Number: 2

```



Here, we choose *_2 - Data For Classification_* and press the Enter key to move forward.



Now, you should see the output below on your screen:



```bash

Successfully loading the built-in training data set

'Data_Classification.xlsx'.

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

--------------------

(Press Enter key to move forward.)

```



We hit Enter key to keep moving.



Then, we choose *_2 - Data For Classification_* as our ***\*Built-in Application Data\****:



```bash

-*-*- Built-in Application Data Option-*-*-

1 - Data For Regression

2 - Data For Classification

3 - Data For Clustering

4 - Data For Dimensional Reduction

(User) ➜ @Number: 2

```



After this, the program will display a list for Column Name:



```bash

Successfully loading the built-in inference data set

'InferenceData_Classification.xlsx'.

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



After pressing the Enter key, you can choose whether you need a world map projection for a specific element option:



```bash

-*-*- World Map Projection -*-*-

World Map Projection for A Specific Element Option:

1 - Yes

2 - No

(Plot) ➜ @Number:

```



More information of the map projection can be found in the section of [World Map Projection](https://geochemistrypi.readthedocs.io/en/latest/For%20User/Model%20Example/Data_Preprocessing/Data%20Preprocessing.html#world-map-projection). In this tutorial, we skip it by typing **2** and pressing the Enter key.



Based on the output prompted, we include column 3 (Label) because it represents the classification label. In a classification task, our goal is to predict or classify data points into specific categories or classes, and the "Label" column contains the information that we want to predict or classify. Then, we also include column 8, 9, 10, 11, 12, 13 (i.e. [8, 13]) in our example.



```bash

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

--------------------

Select the data range you want to process.

Input format:

Format 1: "[**, **]; **; [**, **]", such as "[1, 3]; 7; [10, 13]" --> you want to deal with the columns 1, 2, 3, 7, 10, 11, 12, 13

Format 2: "xx", such as "7" --> you want to deal with the columns 7

@input: 3; [8, 13]

```



Have a double-check on your selection and press Enter to move forward:

```

--------------------

Index - Column Name

3 - Label

8 - SIO2(WT%)

9 - TIO2(WT%)

10 - AL2O3(WT%)

11 - CR2O3(WT%)

12 - FEOT(WT%)

13 - CAO(WT%)

--------------------

(Press Enter key to move forward.)

```



```

The Selected Data Set:

   Label SIO2(WT%) ... FEOT(WT%)  CAO(WT%)

0     1 53.640000 ... 11.130000 20.240000

1     1 52.740000 ... 12.140000 20.480000

2     1 51.710000 ...  6.850000 22.420000

3     1 50.870000 ...  7.530000 22.450000

4     1 50.920000 ...  6.930000 22.620000

...   ...    ... ...    ...    ...

2006   0 52.628866 ...  2.202400 21.172240

2007   0 52.535656 ...  2.093113 21.150105

2008   0 52.163411 ...  2.202465 21.600643

2009   0 44.940000 ...  6.910000 22.520000

2010   0 46.750000 ...  7.550000 22.540000



[2011 rows x 7 columns]

(Press Enter key to move forward.)

```



Now, you should see



```

-*-*- Basic Statistical Information -*-*-

<class 'pandas.core.frame.DataFrame'>

RangeIndex: 2011 entries, 0 to 2010

Data columns (total 7 columns):

 #  Column   Non-Null Count Dtype

--- ------   -------------- -----

 0  Label    2011 non-null  int64

 1  SIO2(WT%)  2010 non-null  float64

 2  TIO2(WT%)  2010 non-null  float64

 3  AL2O3(WT%) 2010 non-null  float64

 4  CR2O3(WT%) 2011 non-null  float64

 5  FEOT(WT%)  2011 non-null  float64

 6  CAO(WT%)  2011 non-null  float64

dtypes: float64(6), int64(1)

memory usage: 110.1 KB

None

Some basic statistic information of the designated data set:

​      Label  SIO2(WT%) ...  FEOT(WT%)   CAO(WT%)

count 2011.00000 2010.000000 ... 2011.000000 2011.000000

mean   0.73446  52.110238 ...   3.215889  21.442025

std    0.44173   2.113287 ...   1.496576   2.325046

min    0.00000   0.218000 ...   1.281000   0.097000

25%    0.00000  51.350135 ...   2.535429  20.532909

50%    1.00000  52.200000 ...   2.920000  21.600000

75%    1.00000  52.980000 ...   3.334500  22.421935

max    1.00000  56.301066 ...  18.270000  26.090000



[8 rows x 7 columns]

Successfully calculate the pair-wise correlation coefficient among

the selected columns.

Save figure 'Correlation Plot' in

/Users/lcthw/geopi/geopi_output/GeoPi - Rock

Classification/XGBoost Algorithm - Test

1/artifacts/image/statistic.

Successfully...

Successfully...

...

Successfully store 'Data Selected' in 'Data Selected.xlsx' in

/Users/lcthw/geopi/geopi_output/GeoPi - Rock

Classification/XGBoost Algorithm - Test 1/artifacts/data.

(Press Enter key to move forward.)

```



You should now see a lot of output on your screen, but don't panic.



This output just provides basic statistical information about the dataset, including count, mean, standard deviation, and percentiles for the data column labeled "Label." It also documents the successful execution of tasks such as calculating correlations, drawing distribution plots, and saving generated charts and data files.



Now, let's press the Enter key to proceed.



```

-*-*- Missing Value Check -*-*-

Check which column has null values:

--------------------

Label     False

SIO2(WT%)   True

TIO2(WT%)   True

AL2O3(WT%)   True

CR2O3(WT%)  False

FEOT(WT%)   False

CAO(WT%)   False

dtype: bool

--------------------

The ratio of the null values in each column:

--------------------

SIO2(WT%)   0.000497

TIO2(WT%)   0.000497

AL2O3(WT%)  0.000497

Label     0.000000

CR2O3(WT%)  0.000000

FEOT(WT%)   0.000000

CAO(WT%)   0.000000

dtype: float64

--------------------

Note: you'd better use imputation techniques to deal with the

missing values.

(Press Enter key to move forward.)

```


## 2. Missing Value Processing



Now, the program will ask us if we want to deal with the missing values, we can choose **yes** here:



```bash

-*-*- Missing Values Process -*-*-

Do you want to deal with the missing values?

1 - Yes

2 - No

(Data) ➜ @Number: 1

```



For strategy, we choose **2 - Impute Missing Values**:



```bash

-*-*- Strategy for Missing Values -*-*-

1 - Drop Rows with Missing Values

2 - Impute Missing Values

Notice: Drop the rows with missing values may lead to a

significant loss of data if too many features are chosen.

Which strategy do you want to apply?

(Data) ➜ @Number: 2

```



Based on the propt, we choose the **1 - Mean Value** in this example and the input data be processed automatically as:



```

-*-*- Imputation Method Option -*-*-

1 - Mean Value

2 - Median Value

3 - Most Frequent Value

4 - Constant(Specified Value)

Which method do you want to apply?

(Data) ➜ @Number: 1

Successfully fill the missing values with the mean value of each

feature column respectively.

(Press Enter key to move forward.)

```



```bash

-*-*- Hypothesis Testing on Imputation Method -*-*-

Null Hypothesis: The distributions of the data set before and

after imputing remain the same.

Thoughts: Check which column rejects null hypothesis.

Statistics Test Method: Kruskal Test

Significance Level: 0.05

The number of iterations of Monte Carlo simulation: 100

The size of the sample for each iteration (half of the whole data

set): 1005

Average p-value:

Label 1.0

SIO2(WT%) 0.9993660077630827

TIO2(WT%) 0.9966146379846705

AL2O3(WT%) 0.9981857963077964

CR2O3(WT%) 1.0

FEOT(WT%) 1.0

CAO(WT%) 1.0

Note: 'p-value < 0.05' means imputation method doesn't apply to

that column.

The columns which rejects null hypothesis: None

Successfully draw the respective probability plot (origin vs.

impute) of the selected columns

Save figure 'Probability Plot' in

/Users/lcthw/geopi/geopi_output/GeoPi - Rock

Classification/XGBoost Algorithm - Test

1/artifacts/image/statistic.

Successfully store 'Probability Plot' in 'Probability Plot.xlsx'

in /Users/lcthw/geopi/geopi_output/GeoPi - Rock

Classification/XGBoost Algorithm - Test

1/artifacts/image/statistic.

<class 'pandas.core.frame.DataFrame'>

RangeIndex: 2011 entries, 0 to 2010

Data columns (total 7 columns):

 #  Column   Non-Null Count Dtype

--- ------   -------------- -----

 0  Label    2011 non-null  float64

 1  SIO2(WT%)  2011 non-null  float64

 2  TIO2(WT%)  2011 non-null  float64

 3  AL2O3(WT%) 2011 non-null  float64

 4  CR2O3(WT%) 2011 non-null  float64

 5  FEOT(WT%)  2011 non-null  float64

 6  CAO(WT%)  2011 non-null  float64

dtypes: float64(7)

memory usage: 110.1 KB

None

Some basic statistic information of the designated data set:

​      Label  SIO2(WT%) ...  FEOT(WT%)   CAO(WT%)

count 2011.00000 2011.000000 ... 2011.000000 2011.000000

mean   0.73446  52.110238 ...   3.215889  21.442025

std    0.44173   2.112761 ...   1.496576   2.325046

min    0.00000   0.218000 ...   1.281000   0.097000

25%    0.00000  51.350271 ...   2.535429  20.532909

50%    1.00000  52.200000 ...   2.920000  21.600000

75%    1.00000  52.980000 ...   3.334500  22.421935

max    1.00000  56.301066 ...  18.270000  26.090000



[8 rows x 7 columns]

Successfully store 'Data Selected Dropped-Imputed' in 'Data

Selected Dropped-Imputed.xlsx' in

/Users/lcthw/geopi/geopi_output/GeoPi - Rock

Classification/XGBoost Algorithm - Test 1/artifacts/data.

(Press Enter key to move forward.)


```



The next step is to select your feature engineering options, for simplicity, we omit the specific operations here. For detailed instructions, please see [here](https://geochemistrypi.readthedocs.io/en/latest/For%20User/Model%20Example/Decomposition/decomposition.html#id6).



```bash

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

--------------------

Feature Engineering Option:

1 - Yes

2 - No

(Data) ➜ @Number: 2

Successfully store 'Data Selected Dropped-Imputed

Feature-Engineering' in 'Data Selected Dropped-Imputed

Feature-Engineering.xlsx' in /Users/lcthw/geopi/geopi_output/GeoPi

- Rock Classification/XGBoost Algorithm - Test 1/artifacts/data.

(Press Enter key to move forward.)

```



## 3. Data Processing



We select **2 - Classification** as our model:



```bash

-*-*- Mode Selection -*-*-

1 - Regression

2 - Classification

3 - Clustering

4 - Dimensional Reduction

(Model) ➜ @Number: 2

(Press Enter key to move forward.)

```



Before we start the classfication model training, we have to specify our X and Y data set. in the example of our selected data set, we take column [2,7] as our X set and column 1 as Y.



```bash

-*-*- Data Segmentation - X Set and Y Set -*-*-

Divide the processing data set into X (feature value) and Y

(target value) respectively.

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

--------------------

The selected X data set:

Select the data range you want to process.

Input format:

Format 1: "[**, **]; **; [**, **]", such as "[1, 3]; 7; [10, 13]" --> you want to deal with the columns 1, 2, 3, 7, 10, 11, 12, 13

Format 2: "xx", such as "7" --> you want to deal with the columns 7

@input: [2, 7]

```



```bash

--------------------

Index - Column Name

2 - SIO2(WT%)

3 - TIO2(WT%)

4 - AL2O3(WT%)

5 - CR2O3(WT%)

6 - FEOT(WT%)

7 - CAO(WT%)

--------------------

Successfully create X data set.

The Selected Data Set:

   SIO2(WT%) TIO2(WT%) ... FEOT(WT%)  CAO(WT%)

0   53.640000  0.400000 ... 11.130000 20.240000

1   52.740000  0.386000 ... 12.140000 20.480000

2   51.710000  0.730000 ...  6.850000 22.420000

3   50.870000  0.780000 ...  7.530000 22.450000

4   50.920000  0.710000 ...  6.930000 22.620000

...     ...    ... ...    ...    ...

2006 52.628866  0.409385 ...  2.202400 21.172240

2007 52.535656  0.422012 ...  2.093113 21.150105

2008 52.163411  0.665545 ...  2.202465 21.600643

2009 44.940000  3.930000 ...  6.910000 22.520000

2010 46.750000  3.360000 ...  7.550000 22.540000



[2011 rows x 6 columns]

Basic Statistical Information:

Some basic statistic information of the designated data set:

​     SIO2(WT%)  TIO2(WT%) ...  FEOT(WT%)   CAO(WT%)

count 2011.000000 2011.000000 ... 2011.000000 2011.000000

mean   52.110238   0.411301 ...   3.215889  21.442025

std    2.112761   0.437225 ...   1.496576   2.325046

min    0.218000   0.000000 ...   1.281000   0.097000

25%   51.350271   0.166500 ...   2.535429  20.532909

50%   52.200000   0.320000 ...   2.920000  21.600000

75%   52.980000   0.511400 ...   3.334500  22.421935

max   56.301066   6.970000 ...  18.270000  26.090000



[8 rows x 6 columns]

Successfully store 'X Without Scaling' in 'X Without Scaling.xlsx'

in /Users/lcthw/geopi/geopi_output/GeoPi - Rock

Classification/XGBoost Algorithm - Test 1/artifacts/data.

(Press Enter key to move forward.)

```



Now, we select our Y data:



```

-*-*- Data Segmentation - X Set and Y Set-*-*-

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

--------------------

The selected Y data set:

Notice: Normally, please choose only one column to be tag column

Y, not multiple columns.

Notice: For classification model training, please choose the label

column which has distinctive integers.

Select the data range you want to process.

Input format:

Format 1: "[**, **]; **; [**, **]", such as "[1, 3]; 7; [10, 13]" --> you want to deal with the columns 1, 2, 3, 7, 10, 11, 12, 13

Format 2: "xx", such as "7" --> you want to deal with the columns 7

@input: 1

```



```bash

--------------------

Index - Column Name

1 - Label

--------------------

Successfully create Y data set.

The Selected Data Set:

   Label

0    1.0

1    1.0

2    1.0

3    1.0

4    1.0

...   ...

2006  0.0

2007  0.0

2008  0.0

2009  0.0

2010  0.0



[2011 rows x 1 columns]

Basic Statistical Information:

Some basic statistic information of the designated data set:

​      Label

count 2011.00000

mean   0.73446

std    0.44173

min    0.00000

25%    0.00000

50%    1.00000

75%    1.00000

max    1.00000

Successfully store 'Y' in 'Y.xlsx' in

/Users/lcthw/geopi/geopi_output/GeoPi - Rock

Classification/XGBoost Algorithm - Test 1/artifacts/data.

(Press Enter key to move forward.)
```



In the following steps, you can choose whether to perform feature scaling and feature selection on your data according to your needs.



```

-*-*- Feature Scaling on X Set -*-*-

1 - Yes

2 - No

(Data) ➜ @Number:

```


```

-*-*- Feature Selection on X set -*-*-

1 - Yes

2 - No

(Data) ➜ @Number:

```



After conducting the two steps above, we come to the point of Data Splitting, in this step, the program is suggesting how to split your dataset into a training set and a test set. Typically, it is recommended to reserve 20% of your entire dataset as the test set. In other words, you should allocate 20% of your data for testing the performance of your model, while the remaining 80% will be used for training the model.



```bash

-*-*- Data Split - Train Set and Test Set -*-*-

Notice: Normally, set 20% of the dataset aside as test set, such

as 0.2.

(Data) ➜ @Test Ratio: 0.2

-------------------------

The Selected Data Set: X Train

   SIO2(WT%) TIO2(WT%) ... FEOT(WT%)  CAO(WT%)

261  54.70400  0.021000 ...  1.595000 24.297000

607  50.51000  0.870000 ...  4.890000 18.940000

1965  53.32721  0.600492 ...  2.076389 21.331999

240  55.09900  0.112000 ...  1.918000 24.274000

819  50.48000  0.490000 ...  4.060000 19.980000

...     ...    ... ...    ...    ...

1130  54.36500  0.053000 ...  2.760000 23.702000

1294  51.17100  0.244000 ...  3.349000 22.085000

860  52.50000  0.360000 ...  3.710000 18.140000

1459  49.04400  0.821000 ...  2.953000 21.859000

1126  51.57800  0.305000 ...  3.397000 21.310000



[1608 rows x 6 columns]

Basic Statistical Information: X Train

Some basic statistic information of the designated data set:

​     SIO2(WT%)  TIO2(WT%) ...  FEOT(WT%)   CAO(WT%)

count 1608.000000 1608.000000 ... 1608.000000 1608.000000

mean   52.113741   0.408489 ...   3.222394  21.465284

std    2.159623   0.426490 ...   1.505620   2.234392

min    0.218000   0.000000 ...   1.354000   0.097000

25%   51.349750   0.172750 ...   2.536859  20.570000

50%   52.170000   0.323500 ...   2.914616  21.600000

75%   52.974750   0.510000 ...   3.334250  22.417750

max   56.294137   6.970000 ...  17.950000  26.090000

[8 rows x 6 columns]

Successfully store 'X Train' in 'X Train.xlsx' in

/Users/lcthw/geopi/geopi_output/GeoPi - Rock

Classification/XGBoost Algorithm - Test 1/artifacts/data.

-------------------------

The Selected Data Set: X Test

   SIO2(WT%) TIO2(WT%) ... FEOT(WT%) CAO(WT%)

1317 53.651000  0.093000 ...  2.74500 23.32500

526  50.360000  1.000000 ...  2.81000 20.86000

393  53.878000  0.187000 ...  3.98200 20.24900

1405 52.714000  0.565000 ...  2.27200 21.17000

433  53.536487  0.371907 ...  2.96938 19.02568

...     ...    ... ...    ...    ...

733  51.780000  0.850000 ...  4.77000 21.32000

1474 54.935000  0.135000 ...  6.56400  1.00000

692  51.800000  0.800000 ...  2.95000 21.80000

1767 54.290000  0.390000 ...  2.52000 21.15000

1624 49.600000  0.580000 ...  3.20000 21.10000

[403 rows x 6 columns]

Basic Statistical Information: X Test

Some basic statistic information of the designated data set:

​    SIO2(WT%)  TIO2(WT%) ...  FEOT(WT%)  CAO(WT%)

count 403.000000 403.000000 ... 403.000000 403.000000

mean  52.096260  0.422518 ...  3.189935  21.349220

std   1.916840  0.478066 ...  1.461479  2.657213

min   40.730000  0.000000 ...  1.281000  0.174000

25%   51.395000  0.147000 ...  2.531162  20.403500

50%   52.229000  0.300000 ...  2.938444  21.590000

75%   52.997000  0.535217 ...  3.333500  22.454500

max   56.301066  5.520000 ...  18.270000  25.171000

[8 rows x 6 columns]

Successfully store 'X Test' in 'X Test.xlsx' in

/Users/lcthw/geopi/geopi_output/GeoPi - Rock

Classification/XGBoost Algorithm - Test 1/artifacts/data.

-------------------------

The Selected Data Set: Y Train

   Label

261   1.0

607   1.0

1965  0.0

240   1.0

819   1.0

...   ...

1130  1.0

1294  1.0

860   1.0

1459  1.0

1126  1.0

[1608 rows x 1 columns]

Basic Statistical Information: Y Train

Some basic statistic information of the designated data set:

​       Label

count 1608.000000

mean   0.730100

std    0.444046

min    0.000000

25%    0.000000

50%    1.000000

75%    1.000000

max    1.000000

Successfully store 'Y Train' in 'Y Train.xlsx' in

/Users/lcthw/geopi/geopi_output/GeoPi - Rock

Classification/XGBoost Algorithm - Test 1/artifacts/data.

-------------------------

The Selected Data Set: Y Test

   Label

1317  1.0

526   1.0

393   1.0

1405  1.0

433   1.0

...   ...

733   1.0

1474  1.0

692   1.0

1767  0.0

1624  0.0

[403 rows x 1 columns]

Basic Statistical Information: Y Test

Some basic statistic information of the designated data set:

​      Label

count 403.000000

mean   0.751861

std   0.432470

min   0.000000

25%   1.000000

50%   1.000000

75%   1.000000

max   1.000000

Successfully store 'Y Test' in 'Y Test.xlsx' in

/Users/lcthw/geopi/geopi_output/GeoPi - Rock

Classification/XGBoost Algorithm - Test 1/artifacts/data.

(Press Enter key to move forward.)

```



## 4. Model Selection



Here, you should see a prompt to ask which model would you like to apply on the dataset you processed in the previous step:



```bash

-*-*- Model Selection -*-*-

1 - Logistic Regression

2 - Support Vector Machine

3 - Decision Tree

4 - Random Forest

5 - Extra-Trees

6 - XGBoost

7 - Multi-layer Perceptron

8 - Gradient Boosting

9 - K-Nearest Neighbors

10 - Stochastic Gradient Descent

11 - All models above to be trained

Which model do you want to apply?(Enter the Corresponding Number)

(Model) ➜ @Number: 6

```



Here, we choose **6 - XGBoost** as our model and let the program employing automated machine learning:



```

Do you want to employ automated machine learning with respect to

this algorithm?(Enter the Corresponding Number):

1 - Yes

2 - No

(Model) ➜ @Number: 1

```



If needed, you can also cutomize your label via the step below, here, we skip this for the moment:



```

-*-*- Customize Label on Label Set -*-*-

1 - Yes

2 - No

(Data) ➜ @Number: 2

```



That's it!



Now, you should see **XGBoost is running**, just sit back and wait for the result:



```

*-**-* XGBoost is running ... *-**-*

Expected Functionality:

+ Model Score

+ Confusion Matrix

+ Cross Validation

+ Model Prediction

+ Model Persistence

+ Precision Recall Curve

+ ROC Curve

+ Two-dimensional Decision Boundary Diagram

+ Permutation Importance Diagram

+ Feature Importance Diagram

[flaml.automl: 02-06 15:02:04] {2599} INFO - task = classification

[flaml.automl: 02-06 15:02:04] {2601} INFO - Data split method: stratified

...

```



Be careful with the path storing your result, you can check it at the end of the output:



```

Successfully store 'Y Test Predict' in 'Y Test Predict.xlsx' in

/Users/lcthw/geopi/geopi_output/GeoPi - Rock

Classification/XGBoost Algorithm - Test 1/artifacts/data.

-----* Model Persistence *-----

Successfully store 'XGBoost' in 'XGBoost.pkl' in

/Users/lcthw/geopi/geopi_output/GeoPi - Rock

Classification/XGBoost Algorithm - Test 1/artifacts/model.

Successfully store 'XGBoost' in 'XGBoost.joblib' in

/Users/lcthw/geopi/geopi_output/GeoPi - Rock

Classification/XGBoost Algorithm - Test 1/artifacts/model.

```
