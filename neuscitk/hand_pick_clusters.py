'''
Script to hand pick clusters from a given dataset of AP waveforms
Written for Python 3.12.4
@ Jeremy Schroeter, August 2024
'''

import sys
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from itertools import cycle


matplotlib.use('TkAgg')
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

def main():
    def onselect(verts):
        path = Path(verts)
        indices = np.nonzero([path.contains_point(xy) for xy in clusters[:, :2]])[0]
        
        if len(indices) > 0:
            # Remove selected points from the main scatter plot
            current_mask[indices] = False
            main_scatter.set_offsets(clusters[current_mask])

            # Create a new scatter plot for the selected points
            color = next(color_cycle)
            new_scatter = ax1.scatter(clusters[indices, 0], clusters[indices, 1], c=[color], label=f'Cluster {current_cluster[0]}')
            scatter_plots.append(new_scatter)

            # ax2.clear()
            ax2.plot(waveforms[indices].T, color=color, alpha=0.1)
            ax2.plot(waveforms[indices].mean(0), color='black', lw=3)

            # Update cluster labels
            cluster_labels[indices] = current_cluster[0]
            current_cluster[0] += 1

            # Update the legend
            ax1.legend()

            fig.canvas.draw_idle()

    # Load embeddings
    embeddings_path, waveforms_path, new_clusters_path = sys.argv[1:4]
    with open(embeddings_path, 'r') as f:
        clusters = json.load(f)
    clusters = np.array(clusters)

    # Load waveforms
    with open(waveforms_path, 'r') as f:
        waveforms = json.load(f)
    waveforms = np.array(waveforms)

    # Initialize cluster labels and mask
    cluster_labels = np.zeros(len(clusters), dtype=int)
    current_cluster = [1]  # Use a list to allow modification inside onselect
    current_mask = np.ones(len(clusters), dtype=bool)

    # Draw plot
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    fig.tight_layout()
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')

    main_scatter = ax1.scatter(clusters[:, 0], clusters[:, 1], c='blue', label='Unassigned')
    scatter_plots = [main_scatter]
    
    # wave_plot = ax2.plot(waveforms.T, color='black', alpha=0.1)

    lasso = LassoSelector(ax1, onselect, props={'c': 'black', 'ls': '--'})

    ax1.legend()
    plt.show()

    # Save cluster labels as JSON
    with open(new_clusters_path, 'w') as f:
        json.dump(cluster_labels.tolist(), f)

if __name__ == '__main__':
    main()