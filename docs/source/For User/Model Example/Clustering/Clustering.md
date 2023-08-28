# Clustering

Cluster analysis itself is not one specific algorithm, but the general task to be solved. It can be achieved by various algorithms that differ significantly in their understanding of what constitutes a cluster and how to efficiently find them. Popular notions of clusters include groups with small distances between cluster members, dense areas of the data space, intervals or particular statistical distributions. Clustering can therefore be formulated as a multi-objective optimization problem. The appropriate clustering algorithm and parameter settings (including parameters such as the distance function to use, a density threshold or the number of expected clusters) depend on the individual data set and intended use of the results. Cluster analysis as such is not an automatic task, but an iterative process of knowledge discovery or interactive multi-objective optimization that involves trial and failure. It is often necessary to modify data preprocessing and model parameters until the result achieves the desired properties.


## 1. Preparation
First, after ensuring the Geochemistry Pi framework has been installed successfully (if not, please see [docs](https://github.com/ZJUEarthData/geochemistrypi/blob/main/docs/source/For%20User/Installation%20Manual.md)), we run the python framework in command line interface to process our program: If you do not input own data, you can run:
```
geochemistrypi data-mining
```
If you prepare to input own data, you can run:
```
geochemistrypi data-mining --data your_own_data_set.xlsx
```
As an example for clustering, at the beginning, we should enter 3:
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
Successfully loading the built-in data set 'Data_Clustering.xlsx'.
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
![Map Projection - AL2O3(WT%)](https://github.com/Darlx/image/raw/main/Map%20Projection%20-%20AL2O3(WT%25).png)
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

It's not necessary to deal with all the data, so in this part, we can choose the data according to our task. In this example, we choose column 8 - SIO2(WT%), 9 - TIO2(WT%), so we enter [8, 9]:
```
Select the data range you want to process.
Input format:
Format 1: "[**, **]; **; [**, **]", such as "[1, 3]; 7; [10, 13]" --> you want to deal with the columns 1, 2, 3, 7, 10, 11, 12, 13
Format 2: "xx", such as "7" --> you want to deal with the columns 7
@input: [8, 9]
After press enter, we can see the chosen data
Index - Column Name
8 - SIO2(WT%)
9 - TIO2(WT%)
--------------------
(Press Enter key to move forward.)
```
And we can see the details of the chosen data and some statistic information, in this process, Correlation Plot, Distribution Histogram and Distribution Histogram After Log Transformation are generated and saved under artifacts\image\statistic folder:
```
The Selected Data Set:
      SIO2(WT%)  TIO2(WT%)
0     53.640000   0.400000
1     52.740000   0.386000
2     51.710000   0.730000
3     50.870000   0.780000
4     50.920000   0.710000
...         ...        ...
2006  52.628866   0.409385
2007  52.535656   0.422012
2008  52.163411   0.665545
2009  44.940000   3.930000
2010  46.750000   3.360000

[2011 rows x 2 columns]
(Press Enter key to move forward.)


Basic Statistical Information:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2011 entries, 0 to 2010
Data columns (total 2 columns):
 #   Column     Non-Null Count  Dtype
---  ------     --------------  -----
 0   SIO2(WT%)  2011 non-null   float64
 1   TIO2(WT%)  2011 non-null   float64
dtypes: float64(2)
memory usage: 31.5 KB
None
Some basic statistic information of the designated data set:
         SIO2(WT%)    TIO2(WT%)
count  2011.000000  2011.000000
mean     52.110416     0.411454
std       2.112777     0.437279
min       0.218000     0.000000
25%      51.350271     0.166500
50%      52.200000     0.320000
75%      52.980000     0.512043
max      56.301066     6.970000
Successfully calculate the pair-wise correlation coefficient among the selected columns.
Save figure 'Correlation Plot' in D:\test\geopi_output\Clustering\example\artifacts\image\statistic.
Successfully store 'Correlation Plot' in 'Correlation Plot.xlsx' in D:\test\geopi_output\Clustering\example\artifacts\image\statistic.
Successfully draw the distribution plot of the selected columns.
Save figure 'Distribution Histogram' in D:\test\geopi_output\Clustering\example\artifacts\image\statistic.
Successfully store 'Distribution Histogram' in 'Distribution Histogram.xlsx' in D:\test\geopi_output\Clustering\example\artifacts\image\statistic.
Successfully draw the distribution plot after log transformation of the selected columns.
Save figure 'Distribution Histogram After Log Transformation' in D:\test\geopi_output\Clustering\example\artifacts\image\statistic.
Successfully store 'Distribution Histogram After Log Transformation' in 'Distribution Histogram After Log Transformation.xlsx' in
D:\test\geopi_output\Clustering\example\artifacts\image\statistic.
Successfully store 'Data Original' in 'Data Original.xlsx' in D:\test\geopi_output\Clustering\example\artifacts\data.
Successfully store 'Data Selected' in 'Data Selected.xlsx' in D:\test\geopi_output\Clustering\example\artifacts\data.
(Press Enter key to move forward.)
```
**Deal with missing value**

After choosing the data, we can use some imputation techniques to deal with the missing value, we can see the values information below:
```
-*-*- Imputation -*-*-
Check which column has null values:
--------------------
SIO2(WT%)    False
TIO2(WT%)    False
dtype: bool
--------------------
The ratio of the null values in each column:
--------------------
SIO2(WT%)    0.0
TIO2(WT%)    0.0
dtype: float64
--------------------
Note: you don't need to deal with the missing values, we'll just pass this step!
(Press Enter key to move forward.)
```
In this example we don’t need to deal with the missing value, so just move forward.

**Feature engineering**

Then, you can construct some features with entering 1.
```
-*-*- Feature Engineering -*-*-
The Selected Data Set:
--------------------
Index - Column Name
1 - SIO2(WT%)
2 - TIO2(WT%)
--------------------
Feature Engineering Option:
1 - Yes
2 - No
(Data) ➜ @Number:
```
After entering 1, the first thing to do is naming our new feature, in this example, just call it new feature. And we also need to build the formula as “a * b”:
```
Selected data set:
a - SIO2(WT%)
b - TIO2(WT%)
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
@input: a * b
(Press Enter key to move forward.)
```
We can see our new feature is presented:
```
Successfully construct a new feature "new feature".
0        21.456000
1        20.357640
2        37.748300
3        39.678600
4        36.153200
           ...
2006     21.545492
2007     22.170701
2008     34.717115
2009    176.614200
2010    157.080000
Name: new feature, Length: 2011, dtype: float64
(Press Enter key to move forward.)
```
And all the features are shown:
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2011 entries, 0 to 2010
Data columns (total 3 columns):
 #   Column       Non-Null Count  Dtype
---  ------       --------------  -----
 0   SIO2(WT%)    2011 non-null   float64
 1   TIO2(WT%)    2011 non-null   float64
 2   new feature  2011 non-null   float64
dtypes: float64(3)
memory usage: 47.3 KB
None
Some basic statistic information of the designated data set:
         SIO2(WT%)    TIO2(WT%)  new feature
count  2011.000000  2011.000000  2011.000000
mean     52.110416     0.411454    21.024405
std       2.112777     0.437279    20.993375
min       0.218000     0.000000     0.000000
25%      51.350271     0.166500     8.647972
50%      52.200000     0.320000    16.677760
75%      52.980000     0.512043    26.674426
max      56.301066     6.970000   358.397400
(Press Enter key to move forward.)
```
After constructing a new feature, we can enter 1 to construct another or enter 2 to move forward, and we enter 2 here:
```
Do you want to continue to construct a new feature?
1 - Yes
2 - No
(Data) ➜ @Number:2
```

## 3. Model Selection
Then we can move forward to next mode, we need to choose the mode here to process our data, in this example, the task is clustering, so we enter 3 here:
```
-*-*- Mode Selection -*-*-
1 - Regression
2 - Classification
3 - Clustering
4 - Dimensional Reduction
(Model) ➜ @Number:
```
This version of geochemistrypi provide 2 clustering models: Kmeans and DBSCN. Both are popular algorithms used for clustering problems. Here we use Kmeans as an example:
```
-*-*- Model Selection -*-*-:
1 - KMeans
2 - DBSCAN
3 - All models above to be trained
Which model do you want to apply?(Enter the Corresponding Number)
(Model) ➜ @Number: 1
```


## 4. Hyper-Parameters Specification

Before starting the training process, you have to specify the number of clusters for our kmeans model, here we enter 5:
```
-*-*- Hyper-parameters Specification -*-*-
Clusters Number: The number of clusters to form as well as the number of centroids to generate.
Designate the clustering number for KMeans in advance, such as 8.
(Model) ➜ Clusters Number: 5
```
And we also need to initialize the centroids, here are two methods, we select k-means++ for the example:
```
Init: Method for initialization of centroids. The centroids represent the center points of the clusters in the dataset.
Please specify the method for initialization of centroids. It is generally recommended to leave it set to k-means++.
1 - k-means++
2 - random
(Model) ➜ @Number:1
```
The number of the max iteration and tolerance should be given too, we set the max iteration at 5, and the tolerance at 0.0005:
```
Max Iter: Maximum number of iterations of the k-means algorithm for a single run.
Please specify the maximum number of iterations of the k-means algorithm for a single run. A good starting range could be between 100 and 500, such as 300.
(Model) ➜ Max Iter: 5
Tolerance: Relative tolerance with regards to inertia to declare convergence.
Please specify the relative tolerance with regards to inertia to declare convergence. A good starting range could be between 0.0001 and 0.001, such as 0.0005.
(Model) ➜ Tolerance:0.0005
```
Finally, we should select the algorithm to use for the computation, here we choose auto:
```
Algorithm: The algorithm to use for the computation.
Please specify the algorithm to use for the computation. It is generally recommended to leave it set to auto.
Auto: selects 'elkan' for dense data and 'full' for sparse data. 'elkan' is generally faster on data with lower dimensionality, while 'full' is faster on data with higher dimensionality
1 - auto
2 - full
3 - elkan
(Model) ➜ @Number:1
```
Then we can start to run the kmeans model with the dataset.


## 5. Results

The clustering result will be printed and saved to the artifacts\data directory. The Silhouette Diagram will be saved under artifacts\image\model_output, in the meantime, the model is saved under artifacts\model, and the hyperparameter is saved under parameters.
```
*-**-* KMeans is running ... *-**-*
Expected Functionality:
+  Cluster Centers
+  Cluster Labels
+  Model Persistence
+  KMeans Score
-----* Clustering Centers *-----
[[50.83575262  0.53704012 51.37279275]
 [52.45089927  0.35095561 52.80185488]
 [45.08534959  1.66686404 46.75221363]
 [54.44439347  0.20042396 54.64481742]
 [ 0.218       0.163       0.381     ]]
-----* Clustering Labels *-----
      clustering result
0                     3
1                     1
2                     1
3                     0
4                     0
...                 ...
2006                  1
2007                  1
2008                  1
2009                  2
2010                  2

[2011 rows x 1 columns]
Successfully store 'KMeans Result' in 'KMeans Result.xlsx' in D:\test\geopi_output\clustering\example\artifacts\data.
Successfully store 'Hyper Parameters - KMeans' in 'Hyper Parameters - KMeans.txt' in D:\test\geopi_output\clustering\example\parameters.
-----* KMeans Scores *-----
Inertia Score:  2252.7808647437128
Calinski Harabasz Score:  3291.499357305765
Silhouette Score:  0.5062976719320229
-----* Clustering Centers *-----
[[50.83575262  0.53704012 51.37279275]
 [52.45089927  0.35095561 52.80185488]
 [45.08534959  1.66686404 46.75221363]
 [54.44439347  0.20042396 54.64481742]
 [ 0.218       0.163       0.381     ]]
-----* Silhouette Diagram *-----
For n_clusters = 5 The average silhouette_score is : 0.5062976719320229
Save figure 'Silhouette Diagram - KMeans' in D:\test\geopi_output\Clustering\example\artifacts\image\model_output.
Successfully store 'Silhouette Diagram - Data With Labels' in 'Silhouette Diagram - Data With Labels.xlsx' in D:\test\geopi_output\Clustering\example\artifacts\image\model_output.
Successfully store 'Silhouette Diagram - Cluster Centers' in 'Silhouette Diagram - Cluster Centers.xlsx' in D:\test\geopi_output\Clustering\example\artifacts\image\model_output.
```

## 6. Two-dimensional graphs of data
We need to select two dimensions of data to draw the plot, we select 1 - SIO2(WT%) and 2 - TIO2(WT%) as example, and the Cluster Two-Dimensional Diagram will be saved under artifacts\image\model_output:
```
-----* 2 Dimensions Data Selection *-----
The software  is going to draw related 2d graphs.
Currently, the data dimension is beyond 2 dimensions.
Please choose 2 dimensions of the data below.
1 - SIO2(WT%)
2 - TIO2(WT%)
3 - new feature
Choose dimension - 1 data:
(Plot) ➜ @Number: 1
1 - SIO2(WT%)
2 - TIO2(WT%)
3 - new feature
Choose dimension - 2 data:
(Plot) ➜ @Number: 2
The Selected Data Dimension:
--------------------
Index - Column Name
1 - SIO2(WT%)
2 - TIO2(WT%)
--------------------
-----* Cluster Two-Dimensional Diagram *-----
Save figure 'Cluster Two-Dimensional Diagram - KMeans' in D:\test\geopi_output\clustering\example\artifacts\image\model_output.
Successfully store 'Cluster Two-Dimensional Diagram - KMeans' in 'Cluster Two-Dimensional Diagram - KMeans.xlsx' in D:\test\geopi_output\clustering\example\artifacts\image\model_output.
```

## 7. Three-dimensional graphs of data
We need to choose three columns of data to draw the plot，so we choose all three data in the example and the Cluster Three-Dimensional Diagram will be saved under artifacts\image\model_output:
```
-----* 3 Dimensions Data Selection *-----
The software is going to draw related 3d graphs.
Currently, the data dimension is beyond 3 dimensions.
Please choose 3 dimensions of the data below.
1 - SIO2(WT%)
2 - TIO2(WT%)
3 - new feature
Choose dimension - 1 data:
(Plot) ➜ @Number: 1
1 - SIO2(WT%)
2 - TIO2(WT%)
3 - new feature
Choose dimension - 2 data:
(Plot) ➜ @Number: 2
1 - SIO2(WT%)
2 - TIO2(WT%)
3 - new feature
Choose dimension - 3 data:
(Plot) ➜ @Number: 3
The Selected Data Dimension:
--------------------
Index - Column Name
1 - SIO2(WT%)
2 - TIO2(WT%)
3 - new feature
--------------------
-----* Cluster Three-Dimensional Diagram *-----
Save figure 'Cluster Three-Dimensional Diagram - KMeans' in D:\test\geopi_output\clustering\example\artifacts\image\model_output.
Successfully store 'Cluster Two-Dimensional Diagram - KMeans' in 'Cluster Two-Dimensional Diagram - KMeans.xlsx' in D:\test\geopi_output\clustering\example\artifacts\image\model_output.
-----* Model Persistence *-----
Successfully store the trained model 'KMeans' in 'KMeans.pkl' in D:\test\geopi_output\clustering\example\artifacts\model.
Successfully store the trained model 'KMeans' in 'KMeans.joblib' in D:\test\geopi_output\clustering\example\artifacts\model.
```
Together with the kmeans clustering result, some related diagrams will also be generated and saved into the artifacts/images/model_output folder.
![Silhouette Diagram - KMeans.png](https://github.com/Darlx/image/raw/main/Silhouette%20Diagram%20-%20KMeans1.png)

<font color=gray size=1><center>Figure 2 Silhouette Diagram - KMeans</center></font>

![Cluster Two-Dimensional Diagram - KMeans.png](https://github.com/Darlx/image/raw/main/Cluster%20Two-Dimensional%20Diagram%20-%20KMeans1.png)

<font color=gray size=1><center>Figure 3 Cluster Two-Dimensional Diagram - KMeans</center></font>

![Cluster Three-Dimensional Diagram - KMeans.png](https://github.com/Darlx/image/raw/main/Cluster%20Three-Dimensional%20Diagram%20-%20KMeans1.png)

<font color=gray size=1><center>Figure 4 Cluster Three-Dimensional Diagram - KMeans</center></font>

The final trained Kmeans models will be saved in the output/trained_models directory.
