# %% load libraries
import numpy as np
import matplotlib.pyplot as plt

# %% lollipop chart
def lollipop(data_, out_f, x_label, y_label, grouping_var):
    # aggregate data
    data =  df.groupby(grouping_var).size()
    # create figure
    fig, ax = plt.subplots(1, figsize=(6,4))
    # iterate over groups
    for key in data.keys():
        plt.plot([key, key], [-1, data[key]], color='k', marker='', lw=1)
        if key < 2021:
            plt.scatter(key, data[key], color='k')
        else:
            plt.scatter(key, data[key], facecolor='w', edgecolor='k')
    # labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # ticks
    min_x, max_x = np.min(data.keys()), np.max(data.keys())
    xticks = np.arange(min_x, max_x + 1, 1)
    xtick_labels = []
    for i in xticks:
        if (i % 2 == 0) & (i < max_x):
            xtick_labels.append('{}'.format(i))
        elif i == max_x:
            xtick_labels.append('In press')
        else:
            xtick_labels.append('')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation='vertical')
    # hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # save plot
    plt.savefig(out_f,
                transparent=True,
                bbox_inches='tight',
                pad_inches=0)
    # show plot
    plt.show()