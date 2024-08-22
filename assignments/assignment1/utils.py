import numpy as np
import matplotlib.patheffects as pe
import matplotlib.pylab as plt


def plot_slice(slice):
    mat = np.zeros((5, 5))
    for idx in np.ndenumerate(slice):
        mat[np.unravel_index(idx[1], (5, 5))] = 1

    fig, ax = plt.subplots()
    ax.matshow(mat, cmap='Greys')
    ax.set_ylabel("Row Index")
    ax.set_title("Column Index")

    plt.gca().set_xticks([x - 0.5 for x in plt.gca().get_xticks()][1:], minor='true')
    plt.gca().set_yticks([y - 0.5 for y in plt.gca().get_yticks()][1:], minor='true')
    plt.grid(which='minor')
    arr = np.arange(25).reshape(5, 5)

    for (i, j), z in np.ndenumerate(arr):
        ax.text(
            j,
            i,
            '{:d}'.format(z),
            ha='center',
            va='center',
            color='black',
            path_effects=[pe.withStroke(linewidth=4, foreground="white")])
