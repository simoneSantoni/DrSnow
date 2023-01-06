# %% load libraries
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# %% data reduction along with k-means clustering
def data_reduction(topic_2_doc, n_clusters):
    reduced_data = PCA(n_components=2).fit_transform(topic_2_doc)
    kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=4)
    kmeans.fit(reduced_data.astype("float"))
    # step size of the mesh
    h = 0.005
    # plot the decision boundary
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    # return
    return reduced_data, kmeans, x_min, x_max, y_min, y_max, xx, yy, Z
