from utils.base import save_fig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, List


def plot_2d_graph(plot_data:pd.DataFrame, plot_index_1:int, plot_index_2:int) -> None:
    """


    """
    print("")
    print("-----* 2D Scatter Plot *-----")

    # Get name list.
    namelist = plot_data.columns.values.tolist()

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('2D Scatter Plot')
    # Check index is Valid.
    if 0 <= plot_index_1 < len(namelist) and 0 <= plot_index_2 < len(namelist):

        pass
    else:
        raise Exception("The Index Is Wrong.")

    plt.xlabel(namelist[plot_index_1])
    plt.ylabel(namelist[plot_index_2])
    pass

def plot_3d_graph(self):
    pass