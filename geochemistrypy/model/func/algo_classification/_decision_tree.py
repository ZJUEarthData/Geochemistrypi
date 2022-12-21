import matplotlib.pyplot as plt
from sklearn import tree

def decision_tree_plot(cls: object, trained_model: any, algorithm_name: str) -> None:
    """
    Draw the  for analysis.

    Parameters
    ----------
    data: pd.DataFrame (n_samples, n_components)
        Data for silhouette.

    trained_model: any
        The algorithm which to be used

    algorithm_name : str
        the name of the algorithm

    References
    ----------
    Plot the result of the algorithm as a tree

    """
    plt.figure()
    y = cls.y
    X = cls.X
    clf = trained_model.fit(X,y)
    tree.plot_tree(clf, filled=True)
    plt.title(f'{algorithm_name} - tree - plot')