# Clustering

Cluster analysis itself is not one specific algorithm, but the general task to be solved. It can be achieved by various algorithms that differ significantly in their understanding of what constitutes a cluster and how to efficiently find them. Popular notions of clusters include groups with small distances between cluster members, dense areas of the data space, intervals or particular statistical distributions. Clustering can therefore be formulated as a multi-objective optimization problem. The appropriate clustering algorithm and parameter settings (including parameters such as the distance function to use, a density threshold or the number of expected clusters) depend on the individual data set and intended use of the results. Cluster analysis as such is not an automatic task, but an iterative process of knowledge discovery or interactive multi-objective optimization that involves trial and failure. It is often necessary to modify data preprocessing and model parameters until the result achieves the desired properties.

## Modle Selection

This version of geochemistrypi provide 2 clustering models: Kmeans and DBSCN. Both of them are popular algorithms used for clustering problems. Here we use Kmeans as an example.

    -*-*- Model Selection -*-*-:
    1 - KMeans
    2 - DBSCAN
    3 - All models above to be trained
    Which model do you want to apply?(Enter the Corresponding Number)
    (Model) ➜ @Number: 1


## Hyper-Parameters Specification

Before starting the training process, you have to specify the number of clusters for our kmeans model:

    -*-*- Hyper-parameters Specification -*-*-
    Clusters Number: The number of clusters to form as well as the number of centroids to generate.
    Designate the clustering number for KMeans in advance, such as 8.
    (Model) ➜ Clusters Number: 5
Then you can start to run the kmeans model with your dataset.

## Results

The clustering reuslt will bu printed and saved to the output/data directory.

```
*-**-* KMeans is running ... *-**-*
Expected Functionality:
+  Cluster Centers
+  Cluster Labels
+  Model Persistence
+  KMeans Score
-----* Clustering Centers *-----
[[5.17101892e+01 4.42308298e-01 5.68227761e+00 8.53091256e-01
  3.00804669e+00 2.09762421e+01 1.58356352e+01 9.05425723e-02
  1.20011397e+00]
 [4.47251667e+01 5.05000000e-02 1.58216667e+00 9.81666667e-02
  9.22933333e+00 4.75750000e-01 4.35605000e+01 1.55416667e-01
  4.71666667e-02]
 [4.98929615e+01 1.25442308e+00 4.07849038e+00 3.22423077e-01
  7.86734904e+00 2.21453654e+01 1.35080481e+01 1.66548077e-01
  6.53663462e-01]
 [2.18000000e-01 1.63000000e-01 4.82230000e+01 1.54210000e+01
  1.54690000e+01 1.09000000e-01 1.94430000e+01 1.38000000e-01
  0.00000000e+00]
 [5.36671411e+01 1.97028342e-01 2.30795788e+00 6.02898618e-01
  2.70559782e+00 2.28514148e+01 1.70905814e+01 7.75967282e-02
  5.38477790e-01]]
-----* Clustering Labels *-----
      clustering result
0                     2
1                     2
2                     2
3                     2
4                     2
...                 ...
2006                  0
2007                  0
2008                  0
2009                  2
2010                  2

[2011 rows x 1 columns]
Successfully store 'KMeans's result' in 'KMeans's result.xlsx' in C:\Users\12396\output\data.
-----* KMeans Scores *-----
Inertia Score:  15900.688313635477
Calinski Harabasz Score:  1049.6578159713506
Silhouette Score:  0.37361004134779147
-----* Clustering Centers *-----
[[5.17101892e+01 4.42308298e-01 5.68227761e+00 8.53091256e-01
  3.00804669e+00 2.09762421e+01 1.58356352e+01 9.05425723e-02
  1.20011397e+00]
 [4.47251667e+01 5.05000000e-02 1.58216667e+00 9.81666667e-02
  9.22933333e+00 4.75750000e-01 4.35605000e+01 1.55416667e-01
  4.71666667e-02]
 [4.98929615e+01 1.25442308e+00 4.07849038e+00 3.22423077e-01
  7.86734904e+00 2.21453654e+01 1.35080481e+01 1.66548077e-01
  6.53663462e-01]
 [2.18000000e-01 1.63000000e-01 4.82230000e+01 1.54210000e+01
  1.54690000e+01 1.09000000e-01 1.94430000e+01 1.38000000e-01
  0.00000000e+00]
 [5.36671411e+01 1.97028342e-01 2.30795788e+00 6.02898618e-01
  2.70559782e+00 2.28514148e+01 1.70905814e+01 7.75967282e-02
  5.38477790e-01]]
-----* Silhouette Diagram *-----
For n_clusters = 5 The average silhouette_score is : 0.37361004134779147
Save figure 'Silhouette Diagram - KMeans' in C:\Users\12396\output\images\model_output.
```
## 2 dimensions graphs of data
choose two demensions of data to draw the plot.
```
-----* 2 Dimensions Data Selection *-----
The software is going to draw related 2d graphs.
Currently, the data dimension is beyond 2 dimensions.
Please choose 2 columns of the data below.
1 - SIO2(WT%)
2 - TIO2(WT%)
3 - AL2O3(WT%)
4 - CR2O3(WT%)
5 - FEOT(WT%)
6 - CAO(WT%)
7 - MGO(WT%)
8 - MNO(WT%)
9 - NA2O(WT%)
Choose dimension - 1 data:
(Plot) ➜ @Number: 1
1 - SIO2(WT%)
2 - TIO2(WT%)
3 - AL2O3(WT%)
4 - CR2O3(WT%)
5 - FEOT(WT%)
6 - CAO(WT%)
7 - MGO(WT%)
8 - MNO(WT%)
9 - NA2O(WT%)
Choose dimension - 2 data:
(Plot) ➜ @Number: 2
The Selected Data Dimension:
--------------------
Index - Column Name
1 - SIO2(WT%)
2 - TIO2(WT%)
--------------------
-----* Cluster Two-Dimensional Diagram *-----
Save figure 'Cluster Two-Dimensional Diagram - KMeans' in C:\Users\12396\output\images\model_output.
```
## 3 dimensions graphs of data
choose three columns of data to draw the plot.
```
-----* 3 Dimensions Data Selection *-----
The software is going to draw related 3d graphs.
Currently, the data dimension is beyond 3 dimensions.
Please choose 3 dimensions of the data below.
1 - SIO2(WT%)
2 - TIO2(WT%)
3 - AL2O3(WT%)
4 - CR2O3(WT%)
5 - FEOT(WT%)
6 - CAO(WT%)
7 - MGO(WT%)
8 - MNO(WT%)
9 - NA2O(WT%)
Choose dimension - 1 data:
(Plot) ➜ @Number: 2
1 - SIO2(WT%)
2 - TIO2(WT%)
3 - AL2O3(WT%)
4 - CR2O3(WT%)
5 - FEOT(WT%)
6 - CAO(WT%)
7 - MGO(WT%)
8 - MNO(WT%)
9 - NA2O(WT%)
Choose dimension - 2 data:
(Plot) ➜ @Number: 3
1 - SIO2(WT%)
2 - TIO2(WT%)
3 - AL2O3(WT%)
4 - CR2O3(WT%)
5 - FEOT(WT%)
6 - CAO(WT%)
7 - MGO(WT%)
8 - MNO(WT%)
9 - NA2O(WT%)
Choose dimension - 3 data:
(Plot) ➜ @Number: 4
The Selected Data Dimension:
--------------------
Index - Column Name
1 - TIO2(WT%)
2 - AL2O3(WT%)
3 - CR2O3(WT%)
--------------------
-----* Cluster Three-Dimensional Diagram *-----
Save figure 'Cluster Three-Dimensional Diagram - KMeans' in C:\Users\12396\output\images\model_output.
-----* Model Persistence *-----
Successfully store the trained model 'KMeans' in 'KMeans_2023-03-30.pkl' in C:\Users\12396\output\trained_models.
Successfully store the trained model 'KMeans' in 'KMeans_2023-03-30.joblib' in C:\Users\12396\output\trained_models.
```

Together with the kmeans clustering result, some related diagrams will also be generated and saved into the output/images/model_output folder.

![Silhouette Diagram - KMeans.png](https://github.com/ZJUEarthData/geochemistrypi/assets/66779478/3b36dfd4-7d6f-4528-b042-c0caddd8cad5)

<font color=gray size=1><center>Figure 1 Silhouette Diagram - KMeans</center></font>

![Cluster Two-Dimensional Diagram - KMeans.png](https://github.com/ZJUEarthData/geochemistrypi/assets/66779478/fb2ae7fe-fe65-4325-9a49-583edff414c6)

<font color=gray size=1><center>Figure 2 Cluster Two-Dimensional Diagram - KMeans</center></font>

![Cluster Three-Dimensional Diagram - KMeans.png](https://github.com/ZJUEarthData/geochemistrypi/assets/66779478/44f0b80f-a48d-45b8-857e-9a1efbc8d256)

<font color=gray size=1><center>Figure 3 Cluster Three-Dimensional Diagram - KMeans</center></font>

The final trained Kmeans models will be saved in the output/trained_models directory.
