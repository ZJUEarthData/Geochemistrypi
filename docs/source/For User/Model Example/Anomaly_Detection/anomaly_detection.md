#  Anomaly Detection - Isolation Forest

Anomaly detection is a broad problem-solving strategy that encompasses various algorithms, each with its own approach to identifying unusual data points. One such algorithm is the Isolation Forest, which distinguishes itself by constructing an ensemble of decision trees to isolate anomalies. The algorithm's core principle is that anomalies are more easily isolated, requiring fewer splits in the trees compared to normal data points.

The effectiveness of Isolation Forest relies on several key parameters, such as the number of trees in the forest, the splitting strategy, and the way it calculates anomaly scores. These parameters must be carefully chosen to align with the dataset's structure and the specific objectives of the anomaly detection task. For instance, a larger number of trees can improve accuracy, while the choice of splitting criteria can influence the model's sensitivity to different types of anomalies.

The process of anomaly detection using Isolation Forest is iterative and involves an interactive optimization cycle. It begins with data preprocessing, where features are transformed and cleaned to enhance the model's performance. Next, the algorithm is configured with the selected parameters, and the model is trained. The results are then evaluated, and the process is repeated, refining the model and adjusting parameters until the desired level of accuracy and interpretability is achieved.


## Table of Contents

- [1. Train-Test Data Preparation](#1.train-test-data-preparation)
- [2. Missing Value Processing](#2.Missing-Value-Processing)
- [3. Data Processing](#3.Data-Processing)
- [4. Model Selection](#4.Model-Selection)
- [5. Hyper-Parameters Specification](#5.Hyper-Parameters-Specification)
- [6. Results](#6.Results)

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

✨ New Experiment (GeoPi -  Rock Isolation Forest):

  'GeoPi -  Rock Isolation Forest' is activated.

✨ Run Name ( Algorithm - Test 1):

(Press Enter key to move forward.)
```

After pressing the Enter key, the program propts the following options to let you choose the **Built-in Training Data**:

```
-*-*- Built-in Training Data Option-*-*-

1 - Data For Regression

2 - Data For Classification

3 - Data For Clustering

4 - Data For Dimensional Reduction

5 - Data For Anomaly Detection

(User) ➜ @Number: 5
```

Here, we choose *_5 - Data For Anomaly Detection_* and press the Enter key to move forward.

Now, you should see the output below on your screen:

```bash
Successfully loading the built-in training data set

'Data_AnomalyDetection.xlsx'.

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

We hit Enter key to keep moving.

After pressing the Enter key, you can choose whether you need a world map projection for a specific element option:

```bash
-*-*- World Map Projection -*-*-

World Map Projection for A Specific Element Option:

1 - Yes

2 - No

(Plot) ➜ @Number:2

(Press Enter Key to move forward.)
```

More information of the map projection can be found in the section of [World Map Projection](https://geochemistrypi.readthedocs.io/en/latest/For%20User/Model%20Example/Data_Preprocessing/Data%20Preprocessing.html#world-map-projection). In this tutorial, we skip it by typing **2** and pressing the Enter key.

Based on the output prompted, we include column  8, 9, 10, 11, 12, 13, 14, 15, 16 (i.e. [8, 16]) in our example.

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

@input: [8,16]
```

Have a double-check on your selection and press Enter to move forward:

```
--------------------

Index - Column Name

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

```
The Selected Data Set:

     SIO2(WT%)   TIO2(WT%)  ...  MNO(WT%)  NA2O(WT%)

0    53.536000   0.291000   ...  0.083000   0.861000

1    54.160000   0.107000   ...  0.150000   1.411000

2    50.873065   0.720622   ...  0.102185   1.920395

3    52.320156   0.072000   ...  0.078300   1.421235

4    50.504861   0.652259   ...  0.096700   1.822857

...    ...        ...       ...     ...        ...

104  50.980000   2.270000   ...  0.060000   0.640000

105  52.770000   0.480000   ...  0.120000   1.230000

106  54.200000   0.100000   ...  0.130000   1.430000

107  54.560000   0.070000   ...  0.050000   0.960000

108  51.960000   0.550000   ...  0.070000   1.810000


[109 rows x 9 columns]

(Press Enter key to move forward.)

```

Now, you should see

```
-*-*- Basic Statistical Information -*-*-

<class 'pandas.core.frame.DataFrame'>

RangeIndex: 109 entries, 0 to 108

Data columns (total 9 columns):

 #    Column   Non-Null Count     Dtype

--- ------   -------------- -----
 0   SIO2(WT%)   109 non-null    float64

 1   TIO2(WT%)   109 non-null    float64

 2   AL2O3(WT%)  109 non-null    float64

 3   CR2O3(WT%)  98 non-null     float64

 4   FEOT(WT%)   109 non-null    float64

 5   CAO(WT%)    109 non-null    float64

 6   MGO(WT%)    109 non-null    float64

 7   MNO(WT%)    109 non-null    float64

 8   NA2O(WT%)   109 non-null    float64



dtypes: float64(9)

memory usage: 7.8 KB

None

Some basic statistic information of the designated data set:

        SIO2(WT%)   TIO2(WT%)  ...    MNO(WT%)   NA2O(WT%)

count  109.000000  109.000000  ...  109.000000  109.000000

mean    52.407919    0.473108  ...    0.092087    1.150724

std      1.471900    0.776412  ...    0.054002    0.555255

min     44.940000    0.017000  ...    0.000000    0.090000

25%     51.600000    0.150000  ...    0.063075    0.650000

50%     52.340000    0.360000  ...    0.090000    1.220000

75%     53.090000    0.540000  ...    0.110000    1.600000

max     55.509000    6.970000  ...    0.400000    2.224900

[8 rows x 9 columns]

Successfully calculate the pair-wise correlation coefficient among

the selected columns.

Save figure 'Correlation Plot' in

/Users/geopi/geopi_output/GeoPi - Rock

Isolation Forest/ Algorithm - Test

1/artifacts/image/statistic.

Successfully...

Successfully...

...

Successfully store 'Data Selected' in 'Data Selected.xlsx' in

/Users/geopi/geopi_output/GeoPi - Rock

Isolation Forest/Algorithm - Test 1/artifacts/data.

(Press Enter key to move forward.)
```

You should now see a lot of output on your screen, but don't panic.

This output just provides basic statistical information about the dataset, including count, mean, standard deviation, and percentiles for the data column labeled "Label." It also documents the successful execution of tasks such as calculating correlations, drawing distribution plots, and saving generated charts and data files.

Now, let's press the Enter key to proceed.

```bath
-*-*- Missing Value Check -*-*-

Check which column has null values:

--------------------

SIO2(WT%)     False

TIO2(WT%)     False

AL2O3(WT%)    False

CR2O3(WT%)     True

FEOT(WT%)     False

CAO(WT%)      False

MGO(WT%)      False

MNO(WT%)      False

NA2O(WT%)     False

dtype: bool

--------------------

The ratio of the null values in each column:

--------------------

CR2O3(WT%)    0.100917

SIO2(WT%)     0.000000

TIO2(WT%)     0.000000

AL2O3(WT%)    0.000000

FEOT(WT%)     0.000000

CAO(WT%)      0.000000

MGO(WT%)      0.000000

MNO(WT%)      0.000000

NA2O(WT%)     0.000000

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

(Press Enter key to move forward.)

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

Null Hypothesis: The distributions of the data set before and after imputing remain the same.

Thoughts: Check which column rejects null hypothesis.

Statistics Test Method: Kruskal Test

Significance Level: 0.05

The number of iterations of Monte Carlo simulation: 100

The size of the sample for each iteration (half of the whole dataset): 54

Average p-value:

SIO2(WT%) 1.0

TIO2(WT%) 1.0

AL2O3(WT%) 1.0

CR2O3(WT%) 0.9327453056346102

FEOT(WT%) 1.0

CAO(WT%) 1.0

MGO(WT%) 1.0

MNO(WT%) 1.0

NA2O(WT%) 1.0

Note: 'p-value < 0.05' means imputation method doesn't apply to that column.

The columns which rejects null hypothesis: None

Successfully draw the respective probability plot (origin vs. impute) of the selected columns

Save figure 'Probability Plot' in /Users/geopi/geopi_output/GeoPi - Rock Isolation Forest/ Algorithm - Test1/artifacts/image/statistic.

Successfully store 'Probability Plot' in 'Probability Plot.xlsx' in /Users/geopi/geopi_output/GeoPi - Rock Isolation Forest/ Algorithm - Test1/artifacts/image/statistic.

<class 'pandas.core.frame.DataFrame'>

RangeIndex: 2011 entries, 0 to 108

Data columns (total 9 columns):

 #  Column      Non-Null Count    Dtype

--- ------      --------------    -----

 0   SIO2(WT%)   109 non-null    float64

 1   TIO2(WT%)   109 non-null    float64

 2   AL2O3(WT%)  109 non-null    float64

 3   CR2O3(WT%)  109 non-null    float64

 4   FEOT(WT%)   109 non-null    float64

 5   CAO(WT%)    109 non-null    float64

 6   MGO(WT%)    109 non-null    float64

 7   MNO(WT%)    109 non-null    float64

 8   NA2O(WT%)   109 non-null    float64

dtypes: float64(9)

memory usage: 7.8 KB

None

Some basic statistic information of the designated data set:

        SIO2(WT%)   TIO2(WT%)  ...    MNO(WT%)   NA2O(WT%)

count  109.000000  109.000000  ...   109.000000  109.000000

mean    52.407919    0.473108  ...    0.092087    1.150724

std      1.471900    0.776412  ...    0.054002    0.555255

min     44.940000    0.017000  ...    0.000000    0.090000

25%     51.600000    0.150000  ...    0.063075    0.650000

50%     52.340000    0.360000  ...    0.090000    1.220000

75%     53.090000    0.540000  ...    0.110000    1.600000

max     55.509000    6.970000  ...    0.400000    2.224900

[8 rows x 9 columns]

Successfully store 'Data Selected Dropped-Imputed' in 'Data Selected Dropped-Imputed.xlsx' in /Users/geopi/geopi_output/GeoPi - Rock Isolation Forest/Algorithm - Test 1/artifacts/data.

(Press Enter key to move forward.)

```



The next step is to select your feature engineering options, for simplicity, we omit the specific operations here. For detailed instructions, please see [here](https://geochemistrypi.readthedocs.io/en/latest/For%20User/Model%20Example/Decomposition/decomposition.html#id6).



```bash

-*-*- Feature Engineering -*-*-

The Selected Data Set:

--------------------

Index - Column Name

1 - SIO2(WT%)

2 - TIO2(WT%)

3 - AL2O3(WT%)

4 - CR2O3(WT%)

5 - FEOT(WT%)

6 - CAO(WT%)

7 - MGO(WT%)

8 - MNO(WT%)

9 - NA2O(WT%)

--------------------

Feature Engineering Option:

1 - Yes

2 - No

(Data) ➜ @Number: 2

Successfully store 'Data Selected Dropped-Imputed Feature-Engineering' in 'Data Selected Dropped-Imputed Feature-Engineering.xlsx' in /Users/geopi/geopi_output/GeoPi-Rock Isolation Forest/Algorithm - Test 1/artifacts/data.

(Press Enter key to move forward.)

```



## 3. Data Processing



We select **5 - Anomaly Detection** as our model:



```bash

-*-*- Mode Selection -*-*-

1 - Regression

2 - Classification

3 - Clustering

4 - Dimensional Reduction

5 - Anomaly Detection

(Model) ➜ @Number: 5
(Press Enter key to move forward.)

```



In the following steps, you can choose whether to perform feature scaling and feature selection on your data according to your needs.



```

-*-*- Feature Scaling on X Set -*-*-

1 - Yes

2 - No

(Data) ➜ @Number:
```


## 4. Model Selection



This version of geochemistrypi only provide one Abnomal Detection models: Isolation Forest. In the future, we may release more models to choose from. Here we use Isolation Forest as an example.



```bash

-*-*- Model Selection -*-*-

 1 - Isolation Forest

 2 - All models above to be trained

Which model do you want to apply?(Enter the Corresponding Number)

(Model) ➜ @Number: 1
```
## 5.Hyper-Parameters Specification

Before initiating the training process for our Isolation Forest model, please specify the following parameters: the number of trees in the ensemble, the level of data contamination, the number of features, and confirm whether bootstrapped samples are employed during the construction of individual trees:

    -*-*- Isolation Forest - Hyper-parameters Specification -*-*-

    N Estimators: The number of trees in the forest.

    Please specify the number of trees in the forest. A good starting
    range could be between 50 and 500, such as 100.

    (Model) ➜ @N Estimators: 100


    Contamination: The amount of contamination of the data set.

    Please specify the contamination of the data set. A good starting range could be between 0.1 and 0.5, such as 0.3.

    (Model) ➜ @Contamination: 0.3


    Max Features: The number of features to draw from X to train each base estimator.

    Please specify the number of features. A good starting range could be between 1 and the total number of features in the dataset.

    (Model) ➜ @Max Features: 1


    Bootstrap: Whether bootstrap samples are used when building trees.

    Bootstrapping is a technique where a random subset of the data is sampled with replacement to create a new dataset ofthe same size as the original. This new dataset is then used to construct a decision tree in the ensemble. If False, the whole dataset is used to build each tree.


    Please specify whether bootstrap samples are used when building trees. It is generally recommended to leave it as True.


    1 - True

    2 - False

    (Model) ➜ @Number: 1


    Max Samples: The number of samples to draw from X_train to train each base estimator.


    Please specify the number of samples. A good starting range could be between 256 and the number of dataset.
    (Model) ➜ @@Max Samples: 256


Then you can start to run the kmeans model with your dataset.


## 6.Results

The Isolation Forest results will be printed to the console and saved in the 'output/data' directory.

```
*-**-* Isolation Forest is running ... *-**-*

Expected Functionality:

Successfully store 'Hyper Parameters - Isolation Forest' in 'Hyper Parameters - Isolation Forest.txt' in Users/geopi/geopi_output/GeoPi-Rock Isolation Forest/Algorithm - Test 1/parameters.


-----* Anomaly Detection Data *-----

    SIO2(WT%)  TIO2(WT%) ... MNO(WT%)  NA2O(WT%)  is_anomaly

0   53.536000   0.291000 ... 0.083000   0.861000      -1

1   54.160000   0.107000 ... 0.150000   1.411000      -1

2   50.873065   0.720622 ... 0.102185   1.920395       1

3   52.320156   0.072000 ... 0.078300   1.421235       1

4   50.504861   0.652259 ... 0.096700   1.822857       1

...        ...         ...         ...        ...        ...

104  50.980000   2.270000 ... 0.060000   0.640000      -1

105  52.770000   0.480000 ... 0.120000   1.230000       1

106  54.200000   0.100000 ... 0.130000   1.430000      -1

107  54.560000   0.070000 ... 0.050000   0.960000       1

108  51.960000   0.550000 ... 0.070000   1.810000       1

[109 rows x 10 columns]

Successfully store 'X Anomaly Detection' in 'X Anomaly Detection.xlsx' in Users/geopi/geopi_output/GeoPi-Rock Isolation Forest/Algorithm - Test 1/data.

-----* Normal Data *-----

    SIO2(WT%)  TIO2(WT%) ... MNO(WT%)  NA2O(WT%)  is_anomaly

2   50.873065   0.720622 ... 0.102185   1.920395       1

3   52.320156   0.072000 ... 0.078300   1.421235       1

4   50.504861   0.652259 ... 0.096700   1.822857       1

5   51.261212   0.832000 ... 0.091200   1.803011       1

6   51.379075   0.572604 ... 0.107026   1.734338       1

...        ...         ...         ...        ...        ...

100  52.490000   0.380000 ... 0.090000   1.460000       1

101  53.300000   0.180000 ... 0.170000   0.640000       1

105  52.770000   0.480000 ... 0.120000   1.230000       1

107  54.560000   0.070000 ... 0.050000   0.960000       1

108  51.960000   0.550000 ... 0.070000   1.810000       1


[76 rows x 10 columns]

Successfully store 'X Normal' in 'X Normal.xlsx' in Users/geopi/geopi_output/GeoPi-Rock Isolation Forest/Algorithm - Test 1/data.

-----* Anomaly Data *-----

    SIO2(WT%)  TIO2(WT%)  ... MNO(WT%)  NA2O(WT%)      is_anomaly

0    53.536000   0.291000 ... 0.083000   0.861000           -1

1    54.160000   0.107000 ... 0.150000   1.411000           -1

7    47.818032   0.190154 ... 0.141183   0.550881           -1

21   54.746000   0.196000 ... 0.016000   0.593000           -1

22   54.885000   0.041000 ... 0.030000   0.578000           -1

23   55.509000   0.029000 ... 0.004000   0.700000           -1

24   55.266000   0.017000 ... 0.046000   0.407000           -1

25   55.497000   0.065000 ... 0.006000   0.227000           -1

26   53.300000   0.410000 ... 0.080000   1.420000           -1

27   52.475000   0.295000 ... 0.266000   0.341000           -1

31   51.100000   0.844000 ... 0.135000   0.635000           -1

34   51.300000   0.685000 ... 0.116000   0.743000           -1

38   51.772929   0.541714 ... 0.082100   1.447214           -1

39   54.920000   0.140000 ... 0.100000   1.090000           -1

45   53.500000   0.280000 ... 0.250000   0.210000           -1

48   51.780000   0.250000 ... 0.090000   0.090000           -1

52   50.680000   1.540000 ... 0.030000   0.720000           -1

53   44.940000   3.930000 ... 0.050000   0.670000           -1

57   51.990000   0.420000 ... 0.090000   0.470000           -1

58   52.880000   0.150000 ... 0.070000   0.890000           -1

74   51.046833   0.926000 ... 0.089000   1.669833           -1

84   54.298039   0.032900 ... 0.050000   0.293237           -1

88   55.491950   0.137300 ... 0.064950   2.224900           -1

89   54.091625   0.111050 ... 0.056300   1.617625           -1

90   54.112375   0.029475 ... 0.056700   1.645900           -1

93   51.420000   6.970000 ... 0.400000   0.510000           -1

95   52.520000   0.090000 ... 0.130000   0.090000           -1

96   53.654321   0.059960 ... 0.157023   0.183443           -1

97   51.473600   0.719200 ... 0.048850   1.776500           -1

102  51.600000   0.710000 ... 0.120000   2.110000           -1

103  54.500000   0.270000 ... 0.070000   0.890000           -1

104  50.980000   2.270000 ... 0.060000   0.640000           -1

106  54.200000   0.100000 ... 0.130000   1.430000           -1
```
Successfully store 'X Anomaly' in 'X Anomaly.xlsx' in Users/geopi/geopi_output/GeoPi-Rock Isolation Forest/Algorithm - Test 1/data.

    -----* Model Persistence *-----
    Successfully store 'Isolation Forest' in 'Isolation Forest.pkl' in Users/geopi/geopi_output/GeoPi-Rock Isolation Forest/Algorithm - Test 1/artifacts/model.

    Successfully store 'Isolation Forest' in 'Isolation Forest.joblib' in Users/geopi/geopi_output/GeoPi-Rock Isolation Forest/Algorithm - Test 1/artifacts/model.

The final trained Isolation Forest models will be saved in the output/trained_models directory.
