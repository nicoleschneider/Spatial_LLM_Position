#Author: Kent O'Sullivan // osullik@umd.edu

#Core Imports
import os

#Library Imports
import pandas as pd

from plotnine import ggplot, aes, theme_classic
from plotnine import geom_point

#User Imports

#Functions

def load_df_from_csv(source_file:os.path)->pd.DataFrame:
    return (pd.read_csv(source_file))

def generate_metric_scatterplot(df:pd.DataFrame):

    scatterplot = (
    ggplot(df, aes(x='example_dist', y='predicted_dist', color='model')) +
    geom_point() +
    theme_classic ()
)



#Main

if __name__ == "__main__":
    source_file = os.path.join("..","results","geospatial_reasoning_llm.csv")
    output_directory = os.path.join("..","paper","figures")

    df = load_df_from_csv(source_file=source_file)

    print(generate_metric_scatterplot(df=df))