# utilities
from pprint import pprint
# standard stuff
import pandas as pd
# a user defined lollipop chart
from companionCode.charts import lollipop
# to parse BibTeX stuff
from companionCode.handleBibs import df_from_bib


# TODO: export separate BibTeX files from the individual folders


# load data
in_f = "manuscript/references/sampled_studies.bib"
df = df_from_bib(in_f)


lollipop(
    df_=df, x_label="Publication year", y_label="Counts of studies", grouping_var="year"
)


df.groupby('year').size()


tw = pd.crosstab(df['journal'], df['year'])
pprint(tw)


out_f = 'manuscript/exhibits/articles_over_time_and_journals.txt'
tw.to_latex(out_f)






