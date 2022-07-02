import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
plt.rcParams['svg.fonttype'] = 'none'
import seaborn as sns


# colors and class-superclass maps
with open(r'..\data\metadata\color_schemes.json') as infile:
    colors = json.load(infile)

animals_used = [1091, 1093, 1060, 1062, 1074, 1092, 1102, 1076, 1082, 1101]
diabetic =     [1076, 1082, 1101]
impaired =     [1060, 1062, 1074, 1092, 1102]
normal =       [1091, 1093]
animal_tol = {
    1076: 'diabetic', 1082: 'diabetic', 1101: 'diabetic', 1060: 'impaired', 1062: 'impaired', 
    1074: 'impaired', 1092: 'impaired', 1102: 'impaired', 1091: 'normal', 1093: 'normal'}
ap = pd.read_excel(r'..\data\metadata\animal_phenotypes.xlsx', index_col=0)
fg = pd.read_csv(r'..\data\metadata\combined_metab_lipid_file_grouping.csv', index_col=0)

# Use data that was sent to collaborators 
data = pd.read_csv(r'..\data\processed\combined_metabolites_data_with_model_params.csv').set_index('i')
data_cols = data.filter(regex='_FBG|_RBG').columns
fbg_cols = data.filter(regex='_FBG').columns
rbg_cols = data.filter(regex='_RBG').columns

qval_sampling = data['qval_sampling']
qval_gtol = data['qval_ogtt']
qval_cross = data['qval_sampling:ogtt']

ogtt_values = ap.loc[ap['lcms_sampled'], 'OGTT (AUC)']
min_ogtt, max_ogtt = min(ogtt_values), max(ogtt_values)

def parse_pval(pval):
    if pval < 0.0001:
        return '****'
    if pval < 0.001:
        return '***'
    if pval < 0.01:
        return '**'
    if pval < 0.05: 
        return '*'
    else:
        return 'ns'

def plot_quant_vs_ogtt(
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