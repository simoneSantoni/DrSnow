# Authors: Mathew Kallada, Andreas Mueller
# License: BSD 3 clause
"""
=========================================
Plot Hierarchical Clustering Dendrogram
=========================================
This example plots the corresponding dendrogram of a hierarchical clustering
using AgglomerativeClustering and the dendrogram method available in scipy.
"""

# %% libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

# %% dendrogram to search for an adequate number of clusters
def plot_dendrogram(model, x_label, y_label, **kwargs):
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    # create figure
    fig, ax = plt.subplots(1, figsize=(12, 4))
    # labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    # plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    # show plot
    plt.show()


# %% plot reduced (PCA) data with clusters superimposed
def plot_reduced_data(
    reduced_data, kmeans_, x_min, x_max, y_min, y_max, xx_, yy_, Z_, unseen_
):
    Z_ = Z_.reshape(xx_.shape)
    plt.figure(1, figsize=(12, 9))
    plt.clf()
    plt.imshow(
        Z_,
        interpolation="nearest",
        extent=(xx_.min(), xx_.max(), yy_.min(), yy_.max()),
        cmap=plt.cm.binary,
        aspect="auto",
        origin="lower",
    )
    plt.plot(
        reduced_data[len(reduced_data) - unseen_ :, 0],
        reduced_data[len(reduced_data) - unseen_ :, 1],
        "r.",
        alpha=0.5,
        markersize=12,
    )
    plt.plot(
        reduced_data[: len(reduced_data) - unseen_, 0],
        reduced_data[: len(reduced_data) - unseen_, 1],
        "b.",
        alpha=0.5,
        markersize=6,
    )
    ## Plot the centroids as a white X
    # centroids = kmeans_.cluster_centers_
    # plt.scatter(
    #    centroids[:, 0],
    #    centroids[:, 1],
    #    marker="o",
    #    s=169,
    #    linewidths=3,
    #    color="orange",
    #    alpha=0.5,
    #    zorder=10,
    # )
    plt.title(
        "K-means clustering on the digits dataset (PCA-reduced data)\n"
        "Centroids are marked with white cross"
    )
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
