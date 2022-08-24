import json
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))  # Add the src directory to path 
# from src.plots import plot_quant_vs_ogtt, volcano
from src.utils import tight_bbox

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib import patches, transforms
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import seaborn as sns 
import plotly.express as px

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['axes.unicode_minus'] = False  # https://stackoverflow.com/questions/43102564/matplotlib-negative-numbers-on-tick-labels-displayed-as-boxes
plt.style.use('seaborn-ticks')  # 'seaborn-ticks'
sns.set_style('ticks')



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


data['signif_interaction'] = data['qval_sampling:ogtt'] < 0.05
data['signif_sampling'] = data['qval_sampling'] < 0.05
gb_means = (data
            .loc[:, data_cols]
            .groupby(fg['bg_type'], axis=1)
            .mean()
           )

data['fasted_mean'] = gb_means['FBG']
data['fed_mean'] = gb_means['RBG']
data['Log2 Fold Change'] = data['fed_mean'] - data['fasted_mean']

data['Fed - Fasted slope'] = data['coef_fed'] - data['coef_fasted']
data['signif_sampling'] = data['qval_sampling'] < 0.05
data['signif_interact'] = data['qval_sampling:ogtt'] < 0.05
data['log_qval_sampling'] = -np.log10(data['qval_sampling'])
data['log_qval_ogtt'] = -np.log10(data['qval_ogtt'])
data['log_qval_sampling:ogtt'] = -np.log10(data['qval_sampling:ogtt'])
data['is_id'] = data['superclass'] != 'Unidentified'

BIG_SCATTER_OUTLINE_SIZE = 48
BIG_SCATTER_SIZE = 23
SMALL_SCATTER_SIZE = 10

def volcano(x, y, df, metab_type, alpha=0.8, ax=None, legend=False, size=None, sizes=None,):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,4), dpi=100)
    sns.scatterplot(
        data=df.loc[(df['superclass'] != 'Unidentified') & (df['Type'] == metab_type)], 
        x=x, y=y, hue='superclass', palette=colors, 
        size=size, sizes=sizes,
#         s=30, linewidth=0.2, edgecolor='gray',
        edgecolor='0.33', linewidth=0.33,
        ax=ax, alpha=alpha, legend=legend)
    sns.scatterplot(
        data=df.loc[(df['superclass'] == 'Unidentified') & (df['Type'] == metab_type)], 
        x=x, y=y, hue='superclass', palette=colors, ax=ax, 
        s=SMALL_SCATTER_SIZE,
        alpha=0.3, zorder=-1, legend=legend)
    ax.ticklabel_format(style='sci', scilimits=(-2, 2))
    ax.set_ylabel('-log10(q-value)')
    if legend:
        ax.legend(loc=(1.01, 0.1), markerscale=1.2)
#     ax.set_title(y)
    ax.axvline(0, linewidth=0.8, c='0.05', zorder=-99)
    ax.axhline(-np.log10(0.05), linewidth=1, c='0.5', zorder=-200)
    sns.despine()
    return ax
    
    
def circle_annotate(text, xy, xytext, ax, ha='center'):
    ax.annotate(text, xy=xy, xytext=xytext, 
                annotation_clip=False, zorder=-10, va='bottom', color='0.5', fontsize=6,
                arrowprops=dict(width=0.6, headwidth=0.6, facecolor='0.5', edgecolor='0.7'))
    ax.scatter(x=xy[0], y=xy[1], edgecolor='0.5', facecolor='white', linewidth=0.6, 
               s=BIG_SCATTER_OUTLINE_SIZE, zorder=-5)


def metab_volcano_plot(ax=None):
    outliers = {
        'm_133': dict(x=-2, y=2, name='Methylhistidine'),
        'm_136': dict(x=-2, y=1.5, name='GHB'),      
        'm_106': dict(x=-2, y=3, name='Anserine'),      
        'm_27' : dict(x=-0.8, y=0, name='Phenylacetylglycine'),   

        'm_54' : dict(x=1.8, y=4, name='Leucine'), 
        'm_103': dict(x=2, y=3, name='Asparagine'),
        'm_9'  : dict(x=2, y=2.5, name='Threonine'), 
        'm_58' : dict(x=1.85, y=2, name='Isoleucine'),
        'm_22' : dict(x=2, y=1, name='Proline'),   
        'm_111': dict(x=2, y=0.5, name='Alanine'),   
        'm_78' : dict(x=1.85, y=1.75, name='Glucose'),   
        'm_7'  : dict(x=0, y=8, name='Trigonelline'),
        'm_61' : dict(x=0.5, y=3, name='Hydrocinnamic\nacid'),
        'm_20' : dict(x=0.5, y=0, name='Quinic\nacid'),}
    df = data.copy()
    df['outlier'] = df.index.isin(outliers)
    df = df.sort_values('outlier')
    if ax is None:
        fig, ax = plt.subplots(dpi=250, facecolor='white')
    volcano(x='Log2 Fold Change', y='log_qval_sampling', df=df, ax=ax,
            metab_type='metabolite', alpha=1, 
            size='outlier', sizes={True: BIG_SCATTER_SIZE, False: SMALL_SCATTER_SIZE})
    for i, row in df.loc[outliers].iterrows():
        x, y = row['Log2 Fold Change'], row['log_qval_sampling']
        if x < 0:
            ha = 'right'
            xtext = x + outliers[i]['x']
        else: 
            ha = 'left'
            xtext = x + outliers[i]['x'] 
        ID = outliers[i].get('name')
        if ID is None:
            ID = f'm/z {round(row["m/z"], 3)}\nRT {round(row["RT"], 1)} min'
        circle_annotate(text=ID, xy=(x, y), xytext=(xtext, y+outliers[i]['y']), ax=ax, ha=ha)
        # ax.annotate(ID, xy=(x, y), xytext=(xtext, y+outliers[i]['y']), 
                    # arrowprops=dict(width=0.6, headwidth=0.6, facecolor='gray', edgecolor='0.7'),
                    # ha=ha, annotation_clip=False, zorder=-10, va='bottom', color='0.5', fontsize=6)
        # ax.scatter(x, y, edgecolor='0.7', facecolor='white', linewidth=1, s=BIG_SCATTER_OUTLINE_SIZE, 
        # zorder=-5)
    ax.tick_params(axis='y', length=0, )
    ax.tick_params(axis='x', length=2, pad=1, labelsize=6)
    ax.set_yticks([])
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    # ax.text(0.5, -0.2, 'Log2 fold change', ha='center', fontsize=6, transform=ax.transAxes)
    ax.text(s='-log10(q-value)', x=-0.05, y=17, ha='right', va='bottom', rotation=90, color='0.25', fontsize=6)
    for tick in range(0, round(ax.get_ylim()[1]), 5):
        ax.text(x=0.05, y=tick, s=tick, ha='left', va='center', zorder=-20, color='0.25', fontsize=6)
    ax.text(ax.get_xlim()[0], -6, 'Higher in fasted', ha='left', fontsize=5)
    ax.text(ax.get_xlim()[1], -6, 'Higher in non-fasted', ha='right', fontsize=5)
    sns.despine(left=True, ax=ax)
    
    
def metab_coef_coef_plot(ax=None):
    df = data.loc[(data['Type'] == 'metabolite') & (data['ID'] != 'Unidentified')].copy()
    df = df.loc[~(df['ID'].isin(['Anhydrohexose', 'Pentose sugar', 'Hexose sugar']))]
    unid = data.loc[(data['Type'] == 'metabolite') & (data['ID'] == 'Unidentified')].copy()
    outliers = {
        'm_4': dict(x=2e-5, y=1e-5),  # uric acid
        'm_27': dict(x=-0.5e-5, y=1.5e-5),  #phenylacetylglycine
        'm_38': dict(x=2e-5, y=2e-5),  # N-Isovalerylglycine
        'm_80': dict(x=2e-5, y=2e-5),  # Ethyl-beta-D-glucuronide
        'm_78': dict(x=1e-5, y=-2e-5),  # glucose
        'm_65': dict(x=-0.5e-5, y=1.5e-5), # hippuric acid
        'm_45': dict(x=-1.5e-5, y=-1e-5), # methylhistidine
        'm_64': dict(x=-1e-5, y=-2e-5),  # His
        'm_105': dict(x=0.5e-5, y=-2e-5),  # Arg
        'm_13': dict(x=-2e-5, y=3e-5), # Taurine
        'm_19': dict(x=-2e-5, y=1.5e-5),  # Ribose
        'm_136': dict(x=2e-5, y=-2.5e-5), # 3-HBA
        'm_72': dict(x=2e-5, y=-2e-5), # guanidinosucc acid
        'm_54': dict(x=2e-5, y=-2e-5), # Leu 
        'm_58': dict(x=2e-5, y=-2e-5), # Isoleu
    #     'm_135': dict(x=-2e-5, y=2e-5), #indoxyl sulfate
        'm_120': dict(x=0.5e-5, y=2e-5), # 8-hydroxy quinoline
        'm_144': dict(x=1e-5, y=1.5e-5), # anhydro glucitol
        'm_20': dict(x=-2e-5, y=2.5e-5),  # quinic acid
        'm_81': dict(x=-0.5e-5, y=-2e-5),  # ergothioneine
        'm_77': dict(x=-2e-5, y=0.5e-5),  # glutamic acid
    }
    df['outlier'] = df.index.isin(outliers)
    df = df.sort_values('outlier')
    if ax is None:
        fig, ax = plt.subplots(dpi=250, facecolor='white')
    # sns.scatterplot(data=unid, x='coef_fed', y='coef_fasted', hue='superclass', palette=colors, ax=ax, 
    #                 alpha=0.3, size=10)
    sns.scatterplot(
        data=df, x='coef_fed', y='coef_fasted', hue='superclass', ax=ax, palette=colors,
            size='outlier', sizes={True: BIG_SCATTER_SIZE, False: SMALL_SCATTER_SIZE}, 
            alpha=1, edgecolor='0.33', linewidth=0.33)
    ax.ticklabel_format(scilimits=(-1, 1))
    for i, row in data.loc[outliers].iterrows():
        x, y = row['coef_fed'], row['coef_fasted']
        xtext = x + outliers[i]['x'] 
        ax.annotate(row['ID'], xy=(x, y), xytext=(xtext, y+outliers[i]['y']), 
                    arrowprops=dict(width=0.6, headwidth=0.6, facecolor='gray', edgecolor='0.7'),
                    ha='center', annotation_clip=False, zorder=-5, va='bottom', color='0.5', fontsize=6)
        ax.scatter(x, y, edgecolor='0.7', facecolor='white', linewidth=1, s=BIG_SCATTER_OUTLINE_SIZE, 
        zorder=-4)

    # fig.text(0.5, 0.03, 'Log2 fold change', ha='center')
    # ax.text(s='-log10(q-value)', x=-0.05, y=17, ha='right', va='bottom', rotation=90,)
    # for tick in range(0, round(ax.get_ylim()[1]), 5):
    #     ax.text(x=0.05, y=tick, s=tick, ha='left', va='center', zorder=-20, )

    ax.set_ylabel('OGTT gluc. AUC regression slope\n(Fasted samples)', fontsize=6)
    ax.set_xlabel('OGTT gluc. AUC regression slope\n(Non-fasted samples)', fontsize=6)

    # Set x and y limits to make a square
    # max_xlim, max_ylim = abs(max(ax.get_xlim())), abs(max(ax.get_ylim()))
    # max_lim = max(max_xlim, max_ylim)
    # ax.set_xlim(-max_lim, max_lim)
    # ax.set_ylim(-max_lim, max_lim)

    ax.axhline(0, color='0.8', zorder=-99)
    ax.axvline(0, color='0.8', zorder=-99)
    
    ax.yaxis.get_offset_text().set_fontsize(5)
    ax.xaxis.get_offset_text().set_fontsize(5)
    
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    
    ax.grid(linewidth=0.2,)
    ax.tick_params(length=0, pad=1, labelsize=6, labelcolor='0.5')
    handles, labels = ax.get_legend_handles_labels()
    handles = handles[1:-3]
    labels = labels[1:-3]
    nhetero = labels.index('Nitrogen heterocycle')
    aad = labels.index('Amino acid derivative')
    labels[nhetero] = 'Nitrogen\nheterocycle'
    labels[aad] = 'Amino acid\nderivative'
    
    legend = ax.legend(handles=handles, labels=labels, loc=(-0.45, 0), 
                       title='Molecule class', fontsize=6, title_fontsize=6,
                       markerscale=0.6, edgecolor='0.5', 
                       frameon=False, handletextpad=0.2, labelspacing=0.2, borderpad=0.1)
    sns.despine(left=True, bottom=True, ax=ax)
    return legend
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    