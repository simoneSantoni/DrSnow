# %% load libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% lollipop chart
def lollipop(df_, x_label, y_label, grouping_var):
    """[summary]

    Args:
        data_ ([type]): [description]
        out_f ([type]): [description]
        x_label ([type]): [description]
        y_label ([type]): [description]
        grouping_var ([type]): [description]
    """
    # aggregate data
    data = df_.groupby(grouping_var).size()
    # create figure
    fig, ax = plt.subplots(1, figsize=(6, 4))
    # iterate over groups
    for key in data.keys():
        if key < 2021:
            plt.plot([key, key], [-1, data[key]], color="k", marker="", lw=1)
            plt.scatter(key, data[key], color="k")
        else:
            plt.plot([key, key], [-1, data[key]], color="grey", marker="", lw=1)
            plt.scatter(key, data[key], facecolor="grey", edgecolor="grey")
    # labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # ticks
    min_x, max_x = np.min(data.keys()), np.max(data.keys())
    xticks = np.arange(min_x, max_x + 1, 1)
    xtick_labels = []
    for i in xticks:
        if (i % 2 == 0) & (i < max_x):
            xtick_labels.append("{}".format(i))
        elif i == max_x:
            xtick_labels.append("In press")
        else:
            xtick_labels.append("")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation="vertical")
    # hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    # show plot
    plt.show()


# %% line chart
def line_chart(x_, y_, x_label, y_label):
    # create figure
    fig, ax = plt.subplots(1, figsize=(6, 4))
    # plot
    ax.plot(x_, y_, color='k')
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
    # show plot
    plt.show()
