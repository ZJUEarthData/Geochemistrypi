# import sys
from global_variable import GEO_IMAGE_PATH
from utils.base import save_fig
import numpy as np
import matplotlib.pyplot as plt
# sys.path.append("..")


# TODO(Sany hecan@mail2.sysu.edu.cn): test and incorporate it into the current framework
def elements_ratio_map(col1, col2, df):
    """ plot the ratio of two different chemical elements

    :param col1: pd.Series, one column in the data set represents one specific elements
    :param col2: pd.Series, one column in the data set represents one specific elements
    :param df: pd.DataFrame, the data set
    """
    # Used to display the minus sign normally
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(dpi=360)
    ax = fig.add_subplot(111)

    # Data processing
    data = df.loc[:, [col1, col2]]
    data = data.reindex(columns=['add', col1, col2], fill_value=1)
    x = data.loc[:, [col1]]
    y = data.loc[:, [col2]]
    xarray = np.array(x)
    yarray = np.array(y)
    xmax = np.max(xarray)
    xmin = np.min(xarray)
    ymin = np.min(yarray)
    ymax = np.max(yarray)

    # Use regression to fit data
    xMat = np.mat(data.iloc[:, :-1].values)
    yMat = np.mat(data.iloc[:, -1].values).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0:
        print('The matrix is singular and cannot be inverted!')
        return
    ws = xTx.I * (xMat.T * yMat)
    yHat = xMat * ws
    plt.plot(xMat[:, 1], yHat, c='black', ls='-')
    ws = ws.tolist()
    plt.scatter(0, 0, marker='o', color='r', edgecolors='r', s=0.1, alpha=0.6,
                label='Positive')
    plt.scatter(0, 0, marker='^', color='b', edgecolors='b', s=0.1, alpha=0.6,
                label='Negative')
    plt.scatter(0, 0, color='gray', edgecolors='gray', alpha=0.01, s=1, lw=1,
                label=('Major elements(n=' + str(len(x)) + ')'))

    for index in range(len(x)):

        n = int(index)
        y = ws[1] * xarray[n] + ws[0]
        m = y * xarray[n]
        k = xarray[n] * yarray[n]

        if k > m:
            plt.scatter(xarray[n], yarray[n], marker='o', color='r', edgecolors='r', s=10, lw=2,
                        alpha=0.6)
        else:
            plt.scatter(xarray[n], yarray[n], marker='^', color='b', edgecolors='b', s=10,
                        alpha=0.3)

    plt.xlabel(col1, size=15)
    plt.ylabel(col2, size=15)
    plt.xlim(0.96*xmin, 1.02*xmax)
    plt.ylim(ymin, ymax)
    plt.legend(edgecolor='gray', facecolor='gray', fontsize='large', loc='upper right', markerscale=8.5)
    plt.text(0.011, 0.004, str(col2) + '/' + str(col1) + '=' + str(ws[1]), transform=ax.transAxes)
    save_fig(f'elements ratio - {col1.name} vs {col2.name}', GEO_IMAGE_PATH)
