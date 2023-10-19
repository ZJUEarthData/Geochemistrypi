# Data Cleaning & Preprocessing
  When we are working on data-mining or machine learning projects, the quality of your results highly depends on the quality of input data. As a result, data cleaning and preprocessing becomes an important step to make sure your input data is neat and balanced. Normally, data scientists will spend a large portion of their working time on data cleaning. However, Geochemistrypi can conduct this process automatically for you, and you just need to follow some simple steps.

Firstly you need to start the geochemistrypi programm via command line instrucitons. Please refer to **Quick Installation** and **Example** to know how to start geochemistrypi. And now we use a classification data file as a sample.
#### Loading Data

By running the start command, there will be a prompt if your dataset is successfully loaded:

    Successfully loading the built-in data set 'Data_Classification.xlsx'.
    --------------------
    Index - Column Name
    1 - CITATION
    2 - SAMPLE NAME
    3 - Label
    ...
    47 - U(PPM)
    --------------------
    (Press Enter key to move forward.)
#### World Map Projection

After successfully loading your data, you will be asked if you would like to plot a world map projection for a specific element:

    World Map Projection for A Specific Element Option:
    1 - Yes
    2 - No

If choosing yes, you will be asked to select an element for mapping. We choose 10-AL2O3 as an example:


![Map_Projection.png](https://github.com/ZJUEarthData/geochemistrypi/assets/66779478/2502c36d-f5f4-4520-9190-badb62b19dec)
<font color=gray size=1><center>Figure 1 distribution map of AL2O3</center></font>


A world map projection for AL2O3 is produced and automatically saved into  the *output/map* directory. After that, you can either choose ***yes*** to continue with another element or ***no*** to exit projection map mode.

####Statistical Summary

Geochemistrypi can conducte basic statistic summary for a specific range of data.

````
Select the data range you want to process.
Input format:
Format 1: "[**, **]; **; [**, **]", such as "[1, 3]; 7; [10, 13]" --> you want to deal with the columns 1, 2, 3, 7, 10, 11, 12, 13
Format 2: "xx", such as "7" --> you want to deal with the columns 7
````
 
We use column 3;[10,16] as an example. column [10,16] are different elements, while column 3 consists of their corresponding lables.

There will be some statistic summary outputs as follows:
1. A table constituted by the selected columns:

        The Selected Data Set:
                Label  AL2O3(WT%)  CR2O3(WT%)  ...   MGO(WT%)  MNO(WT%)  NA2O(WT%)
        0         1    0.140000    0.695000  ...  11.290000    0.2200   2.590000
        1         1    0.060000    0.695000  ...  10.300000    0.5000   2.250000
        2         1    2.930000    0.380000  ...  13.470000    0.2400   1.200000
        3         1    2.870000    0.640000  ...  12.860000    0.1900   1.190000
        4         1    2.900000    0.300000  ...  13.280000    0.2000   1.230000
        ...     ...         ...         ...  ...        ...       ...        ...
        2006      0    5.612482    0.606707  ...  15.056981    0.0456   1.753544
        2007      0    5.384972    1.278862  ...  14.841571    0.0349   1.710571
        2008      0    4.965511    0.667931  ...  14.999107    0.0723   1.741574
        2009      0    8.110000    0.280000  ...  12.170000    0.0500   0.670000
        2010      0    6.640000    0.000000  ...  11.400000    0.1700   0.680000

2. Some basic statistical information of the dataset:

        Basic Statistical Information:
        <class 'pandas.core.frame.DataFrame'>
        RangeIndex: 2011 entries, 0 to 2010
        Data columns (total 8 columns):
        #   Column      Non-Null Count  Dtype
        ---  ------      --------------  -----
        0   Label       2011 non-null   int64
        1   AL2O3(WT%)  2011 non-null   float64
        2   CR2O3(WT%)  2011 non-null   float64
        3   FEOT(WT%)   2011 non-null   float64
        4   CAO(WT%)    2011 non-null   float64
        5   MGO(WT%)    2011 non-null   float64
        6   MNO(WT%)    2011 non-null   float64
        7   NA2O(WT%)   2011 non-null   float64
        dtypes: float64(7), int64(1)
        memory usage: 125.8 KB
        None
        Some basic statistic information of the designated data set:
            Label   AL2O3(WT%)  ...     MNO(WT%)    NA2O(WT%)
        count  2011.00000  2011.000000  ...  2011.000000  2011.000000
        mean      0.73446     4.627858  ...     0.091170     0.974539
        std       0.44173     2.268114  ...     0.051188     0.632556
        min       0.00000     0.010000  ...     0.000000     0.000000
        25%       0.00000     3.531000  ...     0.068500     0.363000
        50%       1.00000     4.923000  ...     0.089299     0.850000
        75%       1.00000     5.921734  ...     0.108000     1.545363
        max       1.00000    48.223000  ...     1.090000     5.920000

        [8 rows x 8 columns]

3. Correlation Matrix, Distribution Histogram, and Distribution Histogram after Log Transformation of the selected data set (saved in */output/images/statistic*)

![Correlation_Plot.png](https://github.com/ZJUEarthData/geochemistrypi/assets/66779478/d61a97cd-7f01-4fde-947e-b89f5b11a345)

<font color=gray size=1><center>Figure 2 Correlation Plot</center></font>

![Distribution_Histogram.png](https://github.com/ZJUEarthData/geochemistrypi/assets/66779478/d316e9fb-5b37-4fde-9402-84b426845cfe)

<font color=gray size=1><center>Figure 3 Distribution Histogra</center></font>

![Distribution Histogram after Log Transformation.png](https://github.com/ZJUEarthData/geochemistrypi/assets/66779478/a1d3773b-d4f3-4f83-9b13-9c2b9183e9a5)

<font color=gray size=1><center>Figure 4 Distribution Histogram after Log Transformation</center></font>

#### Missing Value
Geochemistrypi will generate null value report for the selected dataset:

            -*-*- Imputation -*-*-
        Check which column has null values:
        --------------------
        Label         False
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
        AL2O3(WT%)    0.0
        CR2O3(WT%)    0.0
        FEOT(WT%)     0.0
        CAO(WT%)      0.0
        MGO(WT%)      0.0
        MNO(WT%)      0.0
        NA2O(WT%)     0.0
        dtype: float64
        Note: you don't need to deal with the missing values, we'll just pass this step!
        (Press Enter key to move forward.)
Note that if there is missing value in the dataset, you have to choose a strategy to deal with missing values.


```
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

####Feature Engineering

You can also genereate new features from the selected dataset. In order to do this, you should state the name of generated column. Here we name our new column "new feature", and then you have to identify some operations to generate the new feature. we simply use `b * c + d` (each column corresponds to an alphbetical letter for convinience) the output is as follows:

```
Feature Engineering Option:
1 - Yes
2 - No
(Data) ➜ @Number: 1
Selected data set:
a - Label
b - AL2O3(WT%)
c - CR2O3(WT%)
d - FEOT(WT%)
e - CAO(WT%)
f - MGO(WT%)
g - MNO(WT%)
h - NA2O(WT%)
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
And then it would show you the new feature.

    Successfully construct a new feature "new feature".
    0       11.227300
    1       12.181700
    2        7.963400
    3        9.366800
    4        7.800000
              ...
    2006     5.607531
    2007     8.979749
    2008     5.519085
    2009     9.180800
    2010     7.550000
    Name: new feature, Length: 2011, dtype: float64

A new column called "new feature" is added to the dataset.

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2011 entries, 0 to 2010
Data columns (total 9 columns):
 #   Column       Non-Null Count  Dtype
---  ------       --------------  -----
 0   Label        2011 non-null   int64
 1   AL2O3(WT%)   2011 non-null   float64
 2   CR2O3(WT%)   2011 non-null   float64
 3   FEOT(WT%)    2011 non-null   float64
 4   CAO(WT%)     2011 non-null   float64
 5   MGO(WT%)     2011 non-null   float64
 6   MNO(WT%)     2011 non-null   float64
 7   NA2O(WT%)    2011 non-null   float64
 8   new feature  2011 non-null   float64
dtypes: float64(8), int64(1)
memory usage: 141.5 KB
None
Some basic statistic information of the designated data set:
            Label   AL2O3(WT%)   CR2O3(WT%)    FEOT(WT%)  ...     MGO(WT%)     MNO(WT%)    NA2O(WT%)  new feature
count  2011.00000  2011.000000  2011.000000  2011.000000  ...  2011.000000  2011.000000  2011.000000  2011.000000
mean      0.73446     4.627858     0.756601     3.215889  ...    16.242567     0.091170     0.974539     7.288095
std       0.44173     2.268114     0.581543     1.496576  ...     2.506461     0.051188     0.632556    17.065308
min       0.00000     0.010000     0.000000     1.281000  ...     5.500000     0.000000     0.000000     1.302109
25%       0.00000     3.531000     0.490500     2.535429  ...    15.411244     0.068500     0.363000     5.253093
50%       1.00000     4.923000     0.695000     2.920000  ...    16.180000     0.089299     0.850000     6.878296
75%       1.00000     5.921734     0.912950     3.334500  ...    16.893500     0.108000     1.545363     8.244286
max       1.00000    48.223000    15.421000    18.270000  ...    49.230000     1.090000     5.920000   759.115883

[8 rows x 9 columns]
```

A new .xlsx file with the new feature you just constructed will be saved into the output/data folder.

By now, you have already done all the data preprocessing steps and can solve your problem with powerful models provided by geochemistrypi.
