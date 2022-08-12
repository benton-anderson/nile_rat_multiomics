import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
plt.rcParams['svg.fonttype'] = 'none'
import seaborn as sns

from .utils import parse_p_value, parse_lipid


# LOAD DATA
colors = json.load(open(r'..\data\metadata\color_schemes.json'))
colors['Non-fasted'] = colors['RBG']
colors['Fasted'] = colors['FBG']
compound_superclasses = json.load(open('../data/metadata/compound_superclasses.json', 'r'))
    
data = pd.read_csv(r'../data/processed/combined_metabolites_data_with_model_params.csv').set_index('i')
data_cols = data.filter(regex='_FBG|_RBG').columns
fbg_cols = data.filter(regex='_FBG').columns
rbg_cols = data.filter(regex='_RBG').columns

ap = pd.read_excel(r'..\data\metadata\animal_phenotypes.xlsx', index_col=0)
fg = pd.read_csv(r'..\data\metadata\combined_metab_lipid_file_grouping.csv', index_col=0)

ogtt_values = ap.loc[ap['lcms_sampled'], 'OGTT (AUC)']
min_ogtt, max_ogtt = min(ogtt_values), max(ogtt_values)


def plot_quant_vs_ogtt(df, x, y, palette, xlabel=None, ylabel=None, 
                           animal_lines=True, legend=False,
                           robust=False, ax=None, scatter_kws=None, line_kws=None):
    if ax is None:
        fig, ax = plt.subplots()
    for bg_type in df['bg_type'].unique():
        sns.regplot(data=df.loc[df['bg_type'] == bg_type], x=x, y=y, n_boot=200, robust=robust, 
                    color=palette[bg_type], truncate=False, label=bg_type, scatter_kws=scatter_kws, line_kws=line_kws,
                    ax=ax, seed=1)
    if legend:
        ax.legend()
    if animal_lines:
        for unique_x in df[x].unique():
            ax.axvline(unique_x, color='gray', alpha=0.5, linewidth=0.5)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    sns.despine()
    return ax


def volcano(x, y, df, metab_type, alpha=0.8, ax=None, legend=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,4), dpi=100)
    sns.scatterplot(
        data=df.loc[(df['superclass'] != 'Unidentified') & (df['Type'] == metab_type)], 
        x=x, y=y, hue='superclass', palette=colors, 
#         s=30, linewidth=0.2, edgecolor='gray',
        edgecolor='0.1', linewidth=0.6,
        ax=ax, alpha=alpha, legend=legend)
    sns.scatterplot(
        data=df.loc[(df['superclass'] == 'Unidentified') & (df['Type'] == metab_type)], 
        x=x, y=y, hue='superclass', palette=colors, ax=ax, 
        s=20,
        alpha=0.3, zorder=-10, legend=legend)
    ax.ticklabel_format(style='sci', scilimits=(-2, 2))
    ax.set_ylabel('-log10(q-value)')
    if legend:
        ax.legend(loc=(1.01, 0.1), markerscale=1.2)
#     ax.set_title(y)
    ax.axvline(0, linewidth=1, c='0.5', zorder=-99)
    ax.axhline(-np.log10(0.05), linewidth=1, c='0.5', zorder=-99)
    sns.despine()
    return ax


def fasted_fed_slope(_type, ax=None, alpha=0.8, legend=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,4), dpi=100)
    sns.scatterplot(
        data=data.loc[(data['superclass'] != 'Unidentified') & (data['Type'] == _type)],
        x='coef_fed', y='coef_fasted', hue='superclass', ax=ax, palette=colors, 
#         s=30, linewidth=0.5, edgecolor='gray',
        edgecolor='0.1', linewidth=0.6,
        alpha=alpha, legend=legend)
    sns.scatterplot(
        data=data.loc[(data['superclass'] == 'Unidentified') & (data['Type'] == _type)],
        x='coef_fed', y='coef_fasted', hue='superclass', ax=ax, palette=colors, s=20,
        alpha=0.28, zorder=-10, legend=legend)

    ###### 2 options for making sure the axes are equally scaled to not bias against non-fasted:
    ########## 1. ax.set_aspect('equal') enforces square, but distorts plot
    ########## 2. ylim average +/- 0.5 * xlim range 
    avg_ylim = np.mean([y for y in ax.get_ylim()])
    xlim_range = abs(ax.get_xlim()[0] - ax.get_xlim()[1])
    ax.set_ylim(avg_ylim-0.5*xlim_range, avg_ylim+0.5*xlim_range)
    ax.set_ylabel('Fasted linear regression slope')
    ax.set_xlabel('Non-fasted linear regression slope')
    ax.ticklabel_format(style='sci', scilimits=(-1, 1))
    ax.axvline(0, c='gray', linewidth=0.8, alpha=0.8, zorder=-99)
    ax.axhline(0, c='gray', linewidth=0.8, alpha=0.8, zorder=-99)
    if legend:
        ax.legend(loc=(0.8, 0.05), markerscale=1.2)
    sns.despine()
    return ax


def pvals_plot(x, y, df, metab_type, alpha=0.8, ax=None, legend=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,4), dpi=120)
    sns.scatterplot(
        data=df.loc[(df['is_id'] == True) & (df['Type'] == metab_type)], 
        x=x, y=y, hue='superclass', palette=colors,
        edgecolor='0.1', linewidth=0.6, ax=ax, legend=legend, alpha=alpha)
    sns.scatterplot(
        data=df.loc[(df['is_id'] == False) & (df['Type'] == metab_type)], 
        x=x, y=y, hue='superclass', palette=colors, s=20, ax=ax, legend=legend, alpha=0.3, zorder=-10)
    ax.axhline(-np.log10(0.05), c='gray', alpha=0.3, zorder=-99)
    ax.axvline(-np.log10(0.05), c='gray', alpha=0.3, zorder=-99)
    if legend:
        ax.legend(loc=(1.01, 0.1), markerscale=1.2)
    sns.despine()
    return ax


def plot_quant_vs_ogtt_old(
    feature, data, 
    ax=None, include_info=False, savefig=False, folder_path=None, file_type=None,
    figsize=(7, 5)):
    """
    Make pretty plot of metabolite quant vs. OGTT AUC.
    
    Option to include q-values or not with include_info param
    
    feature = 'l_762'
    data = the whole dataframe read in from the excel sheet
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)  
    df = data.loc[feature, data_cols].to_frame().join(fg[['ogtt', 'bg_type', 'animal']])
    df.rename({feature: 'quant'}, axis=1, inplace=True)
    df['quant'] = df['quant'].astype('float')
        
    for bg_type in ['FBG', 'RBG']:
        sns.regplot(
            data=df.loc[df['bg_type'] == bg_type], x='ogtt', y='quant', 
            color=colors[bg_type], n_boot=200,
            scatter_kws={'s': 60}, ax=ax, seed=1,)
    min_ogtt, max_ogtt = ap.loc[ap['lcms_sampled'], 'OGTT (AUC)'].min(), ap.loc[ap['lcms_sampled'], 'OGTT (AUC)'].max()
    ax.set_xlim(min_ogtt - 1500, max_ogtt + 1500)
    ax.set_xticks(np.arange(20000, 60001, 10000), labels=['20k', '30k', '40k', '50k', '60k'], fontsize=14)
    ax.set_xlabel('OGTT (AUC)', fontsize=14)
    ax.set_ylabel('log2 quant.', fontsize=14)
    ax.set_yticks(ax.get_yticks(), labels=[str(int(x)) for x in ax.get_yticks()], fontsize=14)
    animal_ogtts = ap.loc[ap['lcms_sampled'], 'OGTT (AUC)']
    for animal_ogtt in animal_ogtts:
        ax.axvline(animal_ogtt, c='gray', linewidth=0.8, alpha=0.3, zorder=-10)
    if include_info:    
        legend_title = '< 0.0001 ****\n< 0.001   ***\n< 0.01     **\n< 0.05     *\n> 0.05     ns'
        ax.legend(
            handles=[
                Line2D([0], [0], marker='o', color='w', label='Fed', markerfacecolor=colors['RBG'], markersize=14),
                Line2D([0], [0], marker='o', color='w', label='Fasted', markerfacecolor=colors['FBG'], markersize=14)],
            loc=(0.6, 1.02), title=legend_title, fontsize=12, ncol=1, frameon=False, markerscale=1.2)
        qvals = data.loc[feature, ['qval_sampling', 'qval_ogtt', 
                                   'qval_sampling:ogtt', 'qval_fasted', 'qval_fed']].to_list()
        _type = data.loc[feature, 'Type']
        unique_id = data.loc[feature, 'unique_id']
        _id = data.loc[feature, 'ID']
        rt = data.loc[feature, 'RT']
        mz = data.loc[feature, 'm/z']
        ax.text(x=0.02, y=0.98,
            s=f'Type: {_type.capitalize()}\n'
            f'ID: {_id}\nRT: {round(rt, 2)}\nm/z: {round(mz, 4)}\n'
            f'q-value sampling:     {parse_pval(qvals[0])}\n'  # round(qvals[0], 4)
            f'q-value gluc. tol.:      {parse_pval(qvals[1])}\n'  # round(qvals[1], 4)
            f'q-value interaction:   {parse_pval(qvals[2])}\n'   # round(qvals[2], 4)
            f'q-value fasted coef.:  {parse_pval(qvals[3])}\n'
            f'q-value fed coef.:       {parse_pval(qvals[4])}\n',
            fontsize=12, ha='left', va='bottom', transform=ax.transAxes)  
    sns.despine()
    if savefig:
        file_name = data.loc[feature, 'Type'] + '_' + str(rt) + '_' + str(mz)
        plt.savefig(f'{folder_path}/{file_name}.{file_type}', dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()
    return ax