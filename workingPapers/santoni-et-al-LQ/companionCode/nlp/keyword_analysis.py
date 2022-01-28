# %%
# load libraries
import nltk
import numpy as np
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# load data
df = pd.read_csv("../../data/management_journal_abstracts.csv", low_memory=False)

# %%
# keywords
keywords = [
    "shock",
    "exogenous shock",
    "endogenous shock",
    "jolt",
    "environmental jolt",
]

# %%
# tag items
for keyword in keywords:
    df.loc[:, keyword] = False
    df.loc[df["Abstract"].str.lower().str.contains(keyword), keyword] = True
    df.loc[df["Title"].str.lower().str.contains(keyword), keyword] = True

# %%
# "jolt" occurrences
# data prep
# --+ order data
df.sort_values(["Source title", "Year"], inplace=True)
# encode journal names
le = preprocessing.LabelEncoder()
le.fit(df["Source title"].unique())
# --+ counts of studies by type of shock
# ----+ jolt
gr = df.loc[(df["jolt"] == 1) & (df["environmental jolt"] == False)].groupby(
    ["Source title"], as_index=False
)
jolt = pd.DataFrame(gr.size())
# ----+ environmental jolt
gr = df.loc[(df["jolt"] == 1) & (df["environmental jolt"] == True)].groupby(
    ["Source title"], as_index=False
)
env_jolt = pd.DataFrame(gr.size())
# ----+ data to pass to the chart
data = pd.DataFrame({"Source title": le.classes_},)
data = pd.merge(data, jolt, on="Source title", how="left")
data = pd.merge(data, env_jolt, on="Source title", how="left")
data.fillna(0, inplace=True)
# create stacked bar chart
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1)
ax.barh(
    data.index, data["size_x"], label='Studies alluding to a "jolt"', color="silver"
)
ax.barh(
    data.index,
    width=data["size_y"],
    left=data["size_x"],
    label='Studies alluding to an "environmental jolt"',
    color="firebrick",
)
# labels and ticks
ax.set_xticks([0, 1, 2, 3, 4])
ax.set_xlabel("Counts of studies")
ax.set_ylabel("")
ax.set_yticks(data.index)
ax.set_yticklabels(le.classes_)
# hide the right and top spines
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.yaxis.set_ticks_position("left")
ax.xaxis.set_ticks_position("bottom")
# legend
ax.legend(loc="best")
# plot
plt.show()

# %%
# "shock" keyword occurrences
# data prep
# --+ order data
df.sort_values(["Source title", "Year"], inplace=True)
# encode journal names
le = preprocessing.LabelEncoder()
le.fit(df["Source title"].unique())
# --+ counts of studies by type of shock
# ----+ non exogenous shock
gr = df.loc[(df["shock"] == 1) & (df["exogenous shock"] == False)].groupby(
    ["Source title"], as_index=False
)
shock = pd.DataFrame(gr.size())
# ----+ exogenous shock
gr = df.loc[(df["shock"] == 1) & (df["exogenous shock"] == True)].groupby(
    ["Source title"], as_index=False
)
ex_shock = pd.DataFrame(gr.size())
# ----+ data to pass to the chart
data = pd.DataFrame({"Source title": le.classes_},)
data = pd.merge(data, shock, on="Source title", how="left")
data = pd.merge(data, ex_shock, on="Source title", how="left")
data.fillna(0, inplace=True)
# create stacked bar chart
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1)
ax.barh(
    data.index, data["size_x"], label='Studies alluding to a "shock"', color="silver"
)
ax.barh(
    data.index,
    width=data["size_y"],
    left=data["size_x"],
    label='Studies alluding to an "exogenous shock"',
    color="firebrick",
)
# labels and ticks
ax.set_xlabel("Counts")
ax.set_ylabel("")
ax.set_yticks(data.index)
ax.set_yticklabels(le.classes_)
# hide the right and top spines
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.yaxis.set_ticks_position("left")
ax.xaxis.set_ticks_position("bottom")
# legend
ax.legend(loc="best")
# plot
plt.show()

# %%
# inter-temporal distro of studies alluding to shocks and/or jolts
# data prep
gr = df.loc[df["Year"] > 1999].groupby("Year")
data = gr["shock"].agg(np.sum)
# bar chart
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(1, 1, 1)
plt.bar(x=data.keys(), height=data.values, color="firebrick")
#plt.plot(data.keys(), data.values, color="firebrick", marker="o", markeredgecolor="white")
# labels and ticks
ax.set_xlabel("Years")
ax.set_ylabel("Counts")
# hide the right and top spines
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.yaxis.set_ticks_position("left")
ax.xaxis.set_ticks_position("bottom")
# grid
ax.grid(axis="y", color="white", linestyle="--")
# show plot
plt.show()

# %%
