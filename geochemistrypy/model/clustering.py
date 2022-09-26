# -*- coding: utf-8 -*-
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# from matplotlib.ticker import FixedLocator, FixedFormatter
import numpy as np
from utils.base import save_data
from utils.base import save_fig
from global_variable import MODEL_OUTPUT_IMAGE_PATH
from global_variable import DATASET_OUTPUT_PATH
from ._base import WorkflowBase


class ClusteringWorkflowBase(WorkflowBase):
    """Base class for Cluster.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    def __init__(self):
        super().__init__()

    # TODO: build virtualization in 2D, 3D graph and silhouette plot
    common_function = ['Cluster Centers',
                       'Cluster Labels',
                       'Virtualization in 2D graph',
                       'Virtualization in 3D graph',
                       'Silhouette Plot']
    special_function = None

    def get_cluster_centers(self):
        print("-----* Clustering Centers *-----")
        print(self.model.cluster_centers_)

    def get_labels(self):
        print("-----* Clustering Labels *-----")
        self.X['clustering result'] = self.model.labels_
        print(self.X)
        save_data(self.X, f"{self.naming}", DATASET_OUTPUT_PATH)


    def plot_silhouette_diagram(self, n_clusters: int = 0, ):
        """Draw the silhouette diagram for analysis.

        Parameters
        ----------
        n_clusters: int
            The number of clusters to form as well as the number of centroids to generate.

        References
        ----------
        Silhouette analysis can be used to study the separation distance between the resulting clusters.
        The silhouette plot displays a measure of how close each point in one cluster is to other points in the
        neighboring clusters and thus provides a way to assess parameters like number of clusters visually.
        This measure has a range of [-1, 1].

        https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
        """
        print("")
        print("-----* Silhouette Analysis *-----")
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.

        # I hope the type of self.X = <class 'pandas.core.frame.DataFrame'>
        ax1.set_ylim([0, len(self.X) + (n_clusters + 1) * 10])

        # For example:cluster_labels = [4 4 1 ... 0 0 0]
        cluster_labels = self.X['clustering result']

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(self.X, cluster_labels)
        print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(self.X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(self.X.iloc[:, [0]], self.X.iloc[:, [1]], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = self.model.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

        print("Successfully graph the Silhouette Diagram.")
        save_fig(f"Silhouette Diagram - {self.naming}", MODEL_OUTPUT_IMAGE_PATH)

    def plot_2d_graph(self):
        print("")
        print("-----* 2D Scatter Plot *-----")
        # Get name
        namelist = self.X.columns.values.tolist()

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_title('2D Scatter Plot')
        plt.xlabel(namelist[0])
        plt.ylabel(namelist[1])

        # Step size of the mesh. Decrease to increase the quality of the VQ.
        if len(namelist) == 3:
            h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].
            x_min, x_max = self.X[namelist[0]].min() - 1, self.X[namelist[0]].max() + 1
            y_min, y_max = self.X[namelist[1]].min() - 1, self.X[namelist[1]].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

            print(np.c_[xx.ravel(), yy.ravel()])
            Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            print("WSX",type(Z))
            plt.imshow(
                Z,
                interpolation="nearest",
                extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                cmap=plt.cm.Paired,
                aspect="auto",
                origin="lower",
            )

        plt.plot(self.X[namelist[0]], self.X[namelist[1]], "k.", markersize=2)
        # has wrong
        # c=self.X['clustering result'], marker='o', cmap=plt.cm.Paired)

        # Plot the centroids as a white X
        plt.scatter(
            self.model.cluster_centers_[:, [0]],
            self.model.cluster_centers_[:, [1]],
            marker="x",
            s=169,
            linewidths=3,
            color="w",
            zorder=10,
        )
        plt.xlabel(namelist[0])
        plt.ylabel(namelist[1])
        ax1.scatter(self.X.iloc[:, [0]], self.X.iloc[:, [1]], c=self.X['clustering result'],
                    marker='o', cmap=plt.cm.Paired)
        ax1.scatter(self.model.cluster_centers_[:, [0]], self.model.cluster_centers_[:, [1]],
                    c=list(set(self.X['clustering result'])), marker='o', cmap=plt.cm.Paired, s=60)

        # plt.legend('x1')
        save_fig(f"Scatter Plot - {self.naming}", MODEL_OUTPUT_IMAGE_PATH)

    def plot_3d_graph(self):
        print("")
        print("-----* Plot 3d Graph *-----")
        nameList = self.X.columns.values.tolist()
        fig = plt.figure(figsize=(12, 6), facecolor='w')
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9)

        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(self.X.iloc[:, [0]], self.X.iloc[:, [1]], self.X.iloc[:, [2]], alpha=0.3, c="#FF0000", s=6)
        plt.title('3D Scatter Plot')
        ax.set_xlabel(nameList[0])
        ax.set_ylabel(nameList[1])
        ax.set_zlabel(nameList[1])
        plt.grid(True)

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(self.X.iloc[:, [0]], self.X.iloc[:, [1]], self.X.iloc[:, [2]],
                    c=self.X['clustering result'], s=6, cmap=plt.cm.Paired, edgecolors='none')
        ax2.set_xlabel(nameList[0])
        ax2.set_ylabel(nameList[1])
        ax2.set_zlabel(nameList[1])
        plt.grid(True)
        save_fig(f"Plot 3d Graph - {self.naming}", MODEL_OUTPUT_IMAGE_PATH)


class KMeansClustering(ClusteringWorkflowBase):

    name = "KMeans"
    special_function = ['KMeans Score']

    def __init__(self,
                 n_clusters=8,
                 init="k-means++",
                 n_init=10,
                 max_iter=300,
                 tol=1e-4,
                 verbose=0,
                 random_state=None,
                 copy_x=True,
                 algorithm="auto"):

        super().__init__()
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.algorithm = algorithm
        self.model = KMeans(n_clusters=self.n_clusters,
                            init=self.init,
                            n_init=self.n_init,
                            max_iter=self.max_iter,
                            tol=self.tol,
                            verbose=self.verbose,
                            random_state=self.random_state,
                            copy_x=self.copy_x,
                            algorithm=self.algorithm)
        self.naming = KMeansClustering.name
        """
        
        Parameters
        ----------
        
        n_clusters : int, default=8
            The number of clusters to form as well as the number of
            centroids to generate.
        
        init : {'k-means++', 'random'}, callable or array-like of shape \
                (n_clusters, n_features), default='k-means++'
            Method for initialization:

            'k-means++' : selects initial cluster centers for k-mean
            clustering in a smart way to speed up convergence. See section
            Notes in k_init for more details.

            'random': choose `n_clusters` observations (rows) at random from data
            for the initial centroids.

            If an array is passed, it should be of shape (n_clusters, n_features)
            and gives the initial centers.

            If a callable is passed, it should take arguments X, n_clusters and a
            random state and return an initialization.
        
        n_init : int, default=10
            Number of time the k-means algorithm will be run with different
            centroid seeds. The final results will be the best output of
            n_init consecutive runs in terms of inertia.
            
        max_iter : int, default=300
            Maximum number of iterations of the k-means algorithm for a
            single run.
            
        tol : float, default=1e-4
            Relative tolerance with regards to Frobenius norm of the difference
            in the cluster centers of two consecutive iterations to declare
            convergence.
            
        verbose : int, default=0
            Verbosity mode.
        
        random_state : int, RandomState instance or None, default=None
            Determines random number generation for centroid initialization. Use
            an int to make the randomness deterministic.
            See :term:`Glossary <random_state>`.
            
        copy_x : bool, default=True
            When pre-computing distances it is more numerically accurate to center
            the data first. If copy_x is True (default), then the original data is
            not modified. If False, the original data is modified, and put back
            before the function returns, but small numerical differences may be
            introduced by subtracting and then adding the data mean. Note that if
            the original data is not C-contiguous, a copy will be made even if
            copy_x is False. If the original data is sparse, but not in CSR format,
            a copy will be made even if copy_x is False.
            
        algorithm : {"auto", "full", "elkan"}, default="auto"
            K-means algorithm to use. The classical EM-style algorithm is "full".
            The "elkan" variation is more efficient on data with well-defined
            clusters, by using the triangle inequality. However it's more memory
            intensive due to the allocation of an extra array of shape
            (n_samples, n_clusters).

            For now "auto" (kept for backward compatibility) chooses "elkan" but it
            might change in the future for a better heuristic.

        References
        ----------------------------------------
        Read more in the :ref:`User Guide <k_means>`.
        https://scikit-learn.org/stable/modules/clustering.html#k-means
        """

    def _get_scores(self):
        print("-----* KMeans Scores *-----")
        print("Inertia Score: ", self.model.inertia_)
        print("Calinski Harabasz Score: ", metrics.calinski_harabasz_score(self.X, self.model.labels_))
        print("Silhouette Score: ", metrics.silhouette_score(self.X, self.model.labels_))

    def special_components(self):
        self._get_scores()

class AffinityPropagationClustering(ClusteringWorkflowBase):
    name = "AffinityPropagation"
    special_function = []

    def __init__(self,
                 *,
                 damping=0.5,
                 max_iter=200,
                 convergence_iter=15,
                 copy=True,
                 preference=None,
                 affinity="euclidean",
                 verbose=False,
                 random_state=None,):

        super().__init__()
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.copy = copy
        self.verbose = verbose
        self.preference = preference
        self.affinity = affinity
        self.random_state = random_state
        self.model = AffinityPropagation(damping=self.damping,
                                         max_iter=self.max_iter,
                                         convergence_iter=self.convergence_iter,
                                         copy=self.copy,
                                         verbose=self.verbose,
                                         preference=self.preference,
                                         affinity=self.affinity,
                                         random_state=self.random_state,
                                         )
        self.naming = AffinityPropagationClustering.name

    def _get_scores(self):
        print("Estimated number of clusters: %d" % n_clusters_)
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
        print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
        print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
        print(
            "Adjusted Mutual Information: %0.3f"
            % metrics.adjusted_mutual_info_score(labels_true, labels)
        )
        print(
            "Silhouette Coefficient: %0.3f"
            % metrics.silhouette_score(X, labels, metric="sqeuclidean")
        )

    def special_components(self):
        self._get_scores()

from sklearn.datasets import make_blobs
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(
    n_samples=300, centers=centers, cluster_std=0.5, random_state=0
)

af = AffinityPropagation(preference=-50, random_state=0).fit(X)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)

print("Estimated number of clusters: %d" % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
print(
    "Adjusted Mutual Information: %0.3f"
    % metrics.adjusted_mutual_info_score(labels_true, labels)
)
print(
    "Silhouette Coefficient: %0.3f"
    % metrics.silhouette_score(X, labels, metric="sqeuclidean")
)

import matplotlib.pyplot as plt
from itertools import cycle

plt.close("all")
plt.figure(1)
plt.clf()

colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = X[cluster_centers_indices[k]]
    plt.plot(X[class_members, 0], X[class_members, 1], col + ".")
    plt.plot(
        cluster_center[0],
        cluster_center[1],
        "o",
        markerfacecolor=col,
        markeredgecolor="k",
        markersize=14,
    )
    for x in X[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.show()

class MeanShiftClustering(ClusteringWorkflowBase):
    name = "MeanShift"
    pass

class SpectralClustering(ClusteringWorkflowBase):
    name = "Spectral"
    pass

class WardHierarchicalClustering(ClusteringWorkflowBase):
    name = "WardHierarchical"
    pass

class AgglomerativeClustering(ClusteringWorkflowBase):
    name = "Agglomerative"
    pass

class DBSCANClustering(ClusteringWorkflowBase):

    name = "DBSCAN"
    special_function = []

    def __init__(self,
                 eps=0.5,
                 min_samples=5,
                 metric="euclidean",
                 metric_params=None,
                 algorithm="auto",
                 leaf_size=30,
                 p=None,
                 n_jobs=None,
                 ):

        super().__init__()
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs

        self.model = DBSCAN(eps=self.eps,
                            min_samples=self.min_samples,
                            metric=self.metric,
                            metric_params=self.min_samples,
                            algorithm=self.algorithm,
                            leaf_size=self.leaf_size,
                            p=self.p,
                            n_jobs=self.n_jobs)

    def special_components(self):
        pass

class OPTICSClustering(ClusteringWorkflowBase):
    name = "OPTICS"
    pass

class GaussianMixturesClustering(ClusteringWorkflowBase):
    name = "GaussianMixtures"
    pass

class BIRCHClusteringClustering(ClusteringWorkflowBase):
    name = "BIRCHClustering"
    pass

class BisectingKMeansClustering(ClusteringWorkflowBase):
    name = "BisectingKMeans"
    pass
