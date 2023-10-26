#tools library

import matplotlib.pyplot as plt  # plotting library
from matplotlib import colors
import numpy as np  # work with numeric arrays without labeled axes
plt.rcParams.update({'font.size': 14})

def _get_plot():
    fig, ax = plt.subplots(figsize=(6.5,5.5))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.set_xlabel('ψ_norm')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    return fig, ax

def _add_legend(ax):
    ax.legend(loc="upper right", fontsize="12")