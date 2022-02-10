#!/usr/bin/env python
# coding: utf-8
"""
--------------------------------------------------------------------------------
    charts.py    |     reproduces the charts on the sample of TM studies
--------------------------------------------------------------------------------
Authors  : Simone Santoni, simone.santoni.1@city.ac.uk
Synopsis :
Status   : on going 
"""

# %% libraries
from matplotlib.axis import Axis
import numpy as np
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd


# %% bar chart for qulitative categories


class BarChart:
    """Pre-process and visualize yearly variables' counts"""

    def __init__(self, name_):
        """Initialize the BarChart class
        Args:
            name_ (string): Name for the chart. Used to export the chart.
        """
        self.name = name_

    def add_data(self, df_, var_):
        """Group the raw data around years
        Args:
            df_ (Pandas DF): Raw data with cases nested under categories
            var_ (string): The Pandas DF column around which to group the data
        """
        # --+ group the data
        df_.loc[:, "count"] = 1
        gr = df_.groupby(var_, as_index=False)
        counts = gr["count"].agg(np.sum)
        counts = counts.loc[counts["count"] > 0]
        self.labels = list(counts[var_])
        self.pos = np.arange(0, len(counts))
        self.counts = counts["count"]

    def plot(
        self,
        use_tex,
        axis_label,
        tick_min,
        tick_step,
        color_="black",
        horizontal_=False,
        fig_size=(6, 4),
    ):
        """[summary]

        Args:
            use_tex ([type]): [description]
            axis_label ([type]): [description]
            color_ (str, optional): [description]. Defaults to "black".
            horizontal_ (bool, optional): [description]. Defaults to False.
            fig_size (tuple, optional): [description]. Defaults to (6, 4).
        """
        # backend
        if use_tex == True:
            matplotlib.use("pgf")
        else:
            pass
        # make the figure
        # --+ figure framework
        fig = plt.figure(figsize=fig_size, frameon=True)
        ax = fig.add_subplot(1, 1, 1)
        # --+ plot data
        if horizontal_ == False:
            # plot
            plt.bar(x=self.pos, height=self.counts, color=color_, width=0.5)
            # label
            ax.set_ylabel(axis_label)
            # ticks and tick labels
            ax.set_xticks(self.pos)
            ax.set_xticklabels(self.labels)
            ax.set_yticks(
                np.arange(tick_min, np.max(self.counts), tick_step)
            )
            # grid
            ax.grid(axis="y", color="white", linestyle="--")
        else:
            # plot
            plt.barh(y=self.pos, width=self.counts, color=color_, height=0.5)
            # label
            ax.set_xlabel(axis_label)
            # ticks and tick labels
            ax.set_xticks(
                np.arange(tick_min, np.max(self.counts), tick_step)
            )
            ax.set_yticks(self.pos)
            ax.set_yticklabels(self.labels)
            # grid
            ax.grid(axis="x", color="white")
        # --+ axis options
        # ----+ hide the right and top spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        # --+ tick label assignements
        plt.setp(
            ax.get_yticklabels(),
            rotation=0,
            ha="right",
            va="center",
            rotation_mode="anchor",
        )
        # save plot to file or show
        if use_tex == True:
            plt.savefig(
                "{}.pgf".format(self.name),
                transparent=True,
                bbox_inches="tight",
                pad_inches=0,
            )
            print(
                """
                  PGF is a non-GUI backend -- nothing to display here!
                  
                  To display the chart here, set 'use_tex' = False 
                  """
            )
        else:
            plt.show()


# %% lollipop chart for inter-temporal data


class LongLollipopChart:
    """Pre-process and visualize yearly variables' counts"""

    def __init__(self, name_):
        """Initialize the LollipopChart class
        Args:
            name_ (string): Name for the chart. Used to export the chart.
        """
        self.name = name_

    def add_data(self, df_, var_):
        """Group the raw data around years
        Args:
            df_ (Pandas DF): Raw data with cases nested under categories
            var_ (string): The Pandas DF column around which to group the data
        """
        # --+ group the data
        df_.loc[:, "count"] = 1
        gr = df_.groupby(var_, as_index=False)
        counts = gr["count"].agg(np.sum)
        # --+ get the timespan
        lb, ub = np.min(counts[var_]), np.max(counts[var_])
        span = pd.DataFrame({var_: np.arange(lb, ub + 1, 1, dtype=int)})
        # --+ merge data
        counts = pd.merge(counts, span, on=var_, how="right")
        counts.loc[counts["count"].isnull(), "count"] = 0
        # --+ x and y values to plot
        counts.set_index(var_, inplace=True)
        counts.sort_index(inplace=True)
        self.pos = np.arange(0, len(counts), 1)
        self.counts = counts["count"].values
        self.lb = lb
        self.ub = ub

    def plot(
        self,
        use_tex,
        fig_size=(6, 4),
        color_="black",
        y_axis_label="Counts",
        edge_color="white",
    ):
        """The lollipop chart according to Tufte
        Args:
            data_ (Pandas DF): The grouped data 
            x_label (string): Name of the first axis variable
            y_label (string): Name of the second axis variable
            grouping_var (string): Column of the Pandas DF for grouping data
            cardinal_labels (logical): If True, it reduces the density of first axis tick labels
        """
        # aggregate data
        # create figure
        fig, ax = plt.subplots(1, figsize=fig_size)
        # plot data
        for i, j in zip(self.pos, self.counts):
            plt.plot([i, i], [-1, j], color=color_, marker="", lw=1)
            plt.scatter(i, j, color=color_, edgecolor=edge_color)
        # ticks and tick labels
        # --+ x-axis
        tick_labels = ["%s" % i for i in np.arange(self.lb, self.ub + 1, 1)]
        ax.set_xticks(self.pos)
        ax.set_xticklabels(tick_labels, rotation="vertical")
        ax.xaxis.set_ticks_position("bottom")
        # ax.set_xlabel('Publication year')
        # --+ y-axis
        ax.set_yticks(np.arange(2, np.max(self.counts), 2))
        ax.yaxis.set_ticks_position("left")
        ax.set_ylabel(y_axis_label)
        # --+ hide the right and top spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        # ax.spines["left"].set_visible(False)
        # --+ grid
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", ls="-", color="grey")
        # save plot to file
        if use_tex == True:
            plt.savefig(
                "{}.pgf".format(self.name),
                transparent=True,
                bbox_inches="tight",
                pad_inches=0,
            )
            print(
                """
                  PGF is a non-GUI backend -- nothing to display here!
                  
                  To display the chart here, set 'use_tex' = False 
                  """
            )
        else:
            plt.show()


# %% bar chart for longitudinal data


class LongBarChart:
    """Pre-process and visualize categorical variables' counts"""

    def __init__(self, name_):
        """Initialize the BarChart class
        Args:
            name_ (string): Name for the chart. Used to export the chart.
        """
        self.name = name_

    def add_data(self, df_, var_):
        """Group the raw data around the categorical variables
        Args:
            df_ (Pandas DF): Raw data with cases nested under categories
            var_ (string): The Pandas DF column around which to group the data
        """
        # --+ group the data
        df_.loc[:, "count"] = 1
        gr = df_.groupby(var_, as_index=False)
        counts = gr["count"].agg(np.sum)
        # --+ get the timespan
        lb, ub = np.min(counts[var_]), np.max(counts[var_])
        span = pd.DataFrame({var_: np.arange(lb, ub + 1, 1, dtype=int)})
        # --+ merge data
        counts = pd.merge(counts, span, on=var_, how="right")
        counts.loc[counts["count"].isnull(), "count"] = 0
        # --+ x and y values to plot
        counts.set_index(var_, inplace=True)
        counts.sort_index(inplace=True)
        self.pos = np.arange(0, len(counts), 1)
        self.counts = counts["count"].values
        self.lb = lb
        self.ub = ub

    def plot(
        self,
        use_tex,
        fig_size=(6, 4),
        color_="black",
        y_axis_label="Counts",
        edge_color="white",
    ):
        """Plot counts with a bar chart
        Args:
            use_tex (Boolean): If True, the chart is rendered with the PGF.
                               Else, a the chart is exported as a PDF file.
            y_axis_label (str, optional): Label for the y axis. Defaults to 'Counts'.
            color_ (str, optional): Bars' color. Defaults to "black".
            edge_color (str, optional): Bars' edge color. Defaults to "white".
        """
        # backend
        if use_tex == True:
            matplotlib.use("pgf")
        else:
            pass
        # make the figure
        # --+ figure framework
        fig = plt.figure(figsize=fig_size, frameon=True)
        ax = fig.add_subplot(1, 1, 1)
        # --+ plot data
        ax.bar(self.pos, self.counts, color=color_, edgecolor=edge_color, width=0.6)
        # --+ x-axis
        tick_labels = ["%s" % i for i in np.arange(self.lb, self.ub, 1)]
        tick_labels.append(r"In-press")
        ax.set_xticks(self.pos)
        ax.set_xticklabels(tick_labels, rotation="vertical")
        ax.xaxis.set_ticks_position("bottom")
        # ax.set_xlabel('Publication year')
        # --+ y-axis
        ax.set_yticks(np.arange(2, np.max(self.counts), 2))
        ax.yaxis.set_ticks_position("left")
        ax.set_ylabel(y_axis_label)
        # --+ hide the right and top spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        # ax.spines["left"].set_visible(False)
        # --+ grid
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", ls="-", color="grey")
        # save plot to file
        if use_tex == True:
            plt.savefig(
                "{}.pgf".format(self.name),
                transparent=True,
                bbox_inches="tight",
                pad_inches=0,
            )
            print(
                """
                  PGF is a non-GUI backend -- nothing to display here!
                  
                  To display the chart here, set 'use_tex' = False 
                  """
            )
        else:
            plt.show()
