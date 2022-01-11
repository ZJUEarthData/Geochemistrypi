# -*- coding: utf-8 -*-
import os
from matplotlib import pyplot as plt


def clear_output():
    flag = input("(Press Enter key to move forward.)")
    if flag == '':
        os.system('clear')


def save_fig(fig_name, image_path, tight_layout=True):
    """Run to save pictures

    :param image_path: where to store the image
    :param fig_name: Picture Name
    """
    path = os.path.join(image_path, fig_name + ".png")
    print(f"Save figure '{fig_name}' in {image_path}")
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)