# -*- coding: utf-8 -*-
from sklearn import metrics
from sklearn.cluster import KMeans
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
from .func.algo_clustering._dbscan import DBSCAN_Plot_2D


class ClusteringWorkflowBase(object):

    name = None
    # TODO: build virtualization in 2D, 3D graph and silhouette plot
    common_function = ['Cluster Centers',
                       'Cluster Labels',
                       'Virtualization in 2D graph',
                       'Virtualization in 3D graph',
                       'Silhouette Plot']
    special_function = None

    @classmethod
    def show_info(cls):
        print("*-*" * 2, cls.name, "is running ...", "*-*" * 2)
        print("Expected Functionality:")
        function = cls.common_function + cls.special_function
        for i in range(len(function)):
            print("+ ", function[i])

    def __init__(self):
        self.model = None
        self.X = None
        self.np_X = None
        self.naming = None


    def fit(self,np_X, X,y=None):
        # keep y to be in consistent with the framework
        self.X = X
        self.np_X = np_X
        self.model.fit(X)

    def get_cluster_centers(self):
        print("-----* Clustering Centers *-----")
        print(self.model.cluster_centers_)

    def get_labels(self):
        print("-----* Clustering Labels *-----")
        self.X['clustering result'] = self.model.labels_
        print(self.X)
        save_data(self.X, f"{self.naming}", DATASET_OUTPUT_PATH)


    def plot_silhouette_diagram(self, n_clusters: int = 0):
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

    def _get_scores(self):
        print("-----* KMeans Scores *-----")
        print("Inertia Score: ", self.model.inertia_)
        print("Calinski Harabasz Score: ", metrics.calinski_harabasz_score(self.X, self.model.labels_))
        print("Silhouette Score: ", metrics.silhouette_score(self.X, self.model.labels_))

    def special_components(self):
        self._get_scores()


class DBSCANClustering(ClusteringWorkflowBase):

    name = "DBSCAN"
    special_function = ["DBSCAN_result_plot","DBSCAN_OUTPUT_EXCEL"]

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
                            min_samples=self.min_samples)
        self.naming = DBSCANClustering.name

    def DATA_Prepare(self) -> None:
        db = self.model.fit(self.np_X)
        return db

    def Plot_DBSCAN(self, db) -> None:
        print("-------** Plot_DBSCAN_2D **----------")
        DBSCAN_Plot_2D(self.np_X, db.labels_, db, MODEL_OUTPUT_IMAGE_PATH )


    def special_components(self) -> None:
        db = self.DATA_Prepare()
        self.Plot_DBSCAN(db=db)