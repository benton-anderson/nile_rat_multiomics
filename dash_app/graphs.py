import json
import importlib
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))  # Add the src directory to path
import src.plots

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import plotly.express as px

plt.style.use('../data/metadata/Nilerat_matplotlib_stylesheet.mplstyle')

# Data imports
# colors = json.load(open('../../data/metadata/color_schemes.json'))
# compound_superclasses = json.load(open('../../data/metadata/compound_superclasses.json', 'r'))
# class_abbrevs = json.load(open('../../data/metadata/molec_class_abbrev.json'))

data = pd.read_csv('../data/processed/combined_metabolites_data_with_model_params.csv').set_index('i')
# data_cols = data.filter(regex='_FBG|_RBG').columns
# fbg_cols = data.filter(regex='_FBG').columns
# rbg_cols = data.filter(regex='_RBG').columns
#
# ap = pd.read_excel('../../data/metadata/animal_phenotypes.xlsx', index_col=0)
# fg = pd.read_csv('../../data/metadata/combined_metab_lipid_file_grouping.csv', index_col=0)


df = data.loc[(data['superclass'] != 'Unidentified') & (data['Type'] == 'lipid')].copy()
df['i'] = df.index
test_figure = px.scatter(
    df, x='coef_fed', y='coef_fasted', color='superclass',
    #     color_discrete_map=colors,
    hover_data=['ID', 'i', 'molec_class'], )
