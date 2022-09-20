import json
import os
import sys
import importlib
# sys.path.insert(1, os.path.join(sys.path[0], '..'))  # Add the src directory to path 
from src.utils import tight_bbox, parse_lipid, parse_p_value, add_jitter, shrink_cbar
from src.plots import plot_quant_vs_ogtt


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


# data['signif_interaction'] = data['qval_sampling:ogtt'] < 0.05
# data['signif_sampling'] = data['qval_sampling'] < 0.05
# gb_means = (data
            # .loc[:, data_cols]
            # .groupby(fg['bg_type'], axis=1)
            # .mean()
           # )

# data['fasted_mean'] = gb_means['FBG']
# data['fed_mean'] = gb_means['RBG']
# data['Log2 Fold Change'] = data['fed_mean'] - data['fasted_mean']

# data['Fed - Fasted slope'] = data['coef_fed'] - data['coef_fasted']
# data['signif_sampling'] = data['qval_sampling'] < 0.05
# data['signif_interact'] = data['qval_sampling:ogtt'] < 0.05
# data['log_qval_sampling'] = -np.log10(data['qval_sampling'])
# data['log_qval_ogtt'] = -np.log10(data['qval_ogtt'])
# data['log_qval_sampling:ogtt'] = -np.log10(data['qval_sampling:ogtt'])
# data['is_id'] = data['superclass'] != 'Unidentified'

# l_ids = data.loc[(data['Type'] == 'lipid') & (data['ID'] != 'unknown')].index
# data.loc[l_ids, 'lipid_class']     = data.loc[l_ids, 'ID'].apply(lambda x: parse_lipid(x)[0])
# data.loc[l_ids, 'extra_label']     = data.loc[l_ids, 'ID'].apply(lambda x: parse_lipid(x)[1])
# data.loc[l_ids, 'fa_carbons']      = data.loc[l_ids, 'ID'].apply(lambda x: parse_lipid(x)[2])
# data.loc[l_ids, 'fa_unsat']        = data.loc[l_ids, 'ID'].apply(lambda x: parse_lipid(x)[3])
# data.loc[l_ids, 'fa_carbon:unsat'] = data.loc[l_ids, 'ID'].apply(lambda x: parse_lipid(x)[4])
# data['pval_asterisks']  = data['qval_sampling:ogtt'].apply(lambda x: parse_p_value(x))

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

    ax.set_ylabel('OGTT gluc. AUC regression slope\n(Fasted samples)', fontsize=6)
    ax.set_xlabel('OGTT gluc. AUC regression slope\n(Non-fasted samples)', fontsize=6)

    ax.axhline(0, color='0.5', linewidth=0.8, zorder=-99)
    ax.axvline(0, color='0.5', linewidth=0.8, zorder=-99)
    
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


def lipid_volcano_plot(ax=None):
    outliers = {
        'l_8'  : dict(x=-0.1, y=0.06, name='AC 16:0'),
        'l_5'  : dict(x=-0.1, y=0.1, name='AC 18:1'),
        'l_10' : dict(x=-0.08, y=0.14, name='AC 14:0'),
        'l_4'  : dict(x=-0.01, y=0.15, name='AC 18:2'),
        'l_569': dict(x=-0.02, y=0.06, name='TG 20:5_22:6_22:6'), # TG 64:17
        'l_582': dict(x=-0.1, y=0.0, name='TG 22:6_22:6_22:6'), # TG 66:18
        'l_372': dict(x=-0.1, y=0.01, name='PE 18:0_22:6'),  # PE 40:6
        'l_652': dict(x=-0.1, y=0.01, name='TG 60:12'),
        'l_665': dict(x=-0.1, y=-0.03, name='TG 58:10'),
        'l_603': dict(x=-0.1, y=-0.03, name='TG 60:13'),
        'l_716': dict(x=-0.1, y=-0.03, name='TG 60:13'),

        'l_222': dict(x=0.1, y=0.08, name='PI 16:0_18:1'),  # PI 34:1
        'l_331': dict(x=0.1, y=-0.01, name='PI 38:2'), 
        'l_235': dict(x=0.2, y=-0.04, name='PC 18:2_18:2'), # PC 36:4
        'l_269': dict(x=0.03, y=-0.04, name='Plasmanyl-PC O-34:3'),
        'l_187': dict(x=0.1, y=0.01, name='PC O-34:4'),
        'l_888': dict(x=0.1, y=0.02, name='TG 18:2_18:1_24:1'), # TG 60:4
        'l_862': dict(x=0.1, y=0.0, name='TG 58:4'), # TG 58:4
        'l_893': dict(x=0.15, y=0.01, name='TG 18:2_18:1_22:0'), # TG 58:3
        'l_866': dict(x=0.1, y=0.02, name='TG 56:3'), 
        'l_828': dict(x=0.1, y=0.02, name='TG 56:4'),
        'l_771': dict(x=0.1, y=0.02, name='TG 50:2'),
        'l_323': dict(x=0.15, y=0.05, name='PI 18:1_18:0'),
        'l_241': dict(x=0.05, y=0, name='Plasmanyl-PC O-34:4'),
        'l_0'  : dict(x=-0.12, y=0.28, name='AC 5:0'),
        # 'l_1'  : dict(x=-0.1, y=0.4, name='AC 4:0'),
        'l_2'  : dict(x=-0.15, y=0.15, name='AC 3:0'),
        'l_3'  : dict(x=0, y=0.35, name='AC 2:0'),
    }
    if ax is None:
        fig, ax = plt.subplots()
    df = data.copy()
    df['outlier'] = df.index.isin(outliers)
    df = df.sort_values('outlier')
    volcano(x='Log2 Fold Change', y='log_qval_sampling', df=df, metab_type='lipid', 
            size='outlier', sizes={True: BIG_SCATTER_SIZE, False: SMALL_SCATTER_SIZE}, ax=ax)
                   
    for i, row in data.loc[outliers].iterrows():
        x, y = row['Log2 Fold Change'], row['log_qval_sampling']        
        # https://stackoverflow.com/questions/29107800/python-matplotlib-convert-axis-data-coordinates-systems
        output_coords = ax.transLimits.transform((x, y))
        xtext = output_coords[0] + outliers[i]['x']
        ytext = output_coords[1] + outliers[i]['y']
        annot_text = row['lipid_class'] + ' ' + row['fa_carbon:unsat']
        ax.annotate(
            annot_text, xy=(x, y), xytext=(xtext, ytext), 
            textcoords='axes fraction',
            va=('center' if abs(outliers[i]['y']) < 0.04 else 'bottom'), 
            ha=('right' if x < 0 else 'left'), 
            annotation_clip=False, zorder=-5, color='0.5', fontsize=6,
            arrowprops=dict(width=0.6, headwidth=0.6, facecolor='gray', edgecolor='0.7'))
        ax.scatter(x, y, edgecolor='0.7', facecolor='white', linewidth=1, s=BIG_SCATTER_OUTLINE_SIZE, 
            zorder=-4)
    
    ax.tick_params(axis='y', length=0, )
    ax.tick_params(axis='x', length=2, pad=1)
    ax.set_ylabel(None)
    ax.set_yticks([])
    ax.set_xlabel(None)
    # fig.text(0.5, 0.03, 'Log2 fold change', ha='center')
    ax.text(s='-log10(q-value)', x=-0.05, y=17, ha='right', va='bottom', rotation=90,)
    for tick in range(0, round(ax.get_ylim()[1]), 5):
        ax.text(x=0.05, y=tick, s=tick, ha='left', va='center', zorder=-20, )

    ax.text(ax.get_xlim()[0]-1, -4, 'Higher in fasted', ha='left', va='top', fontsize=5)
    ax.text(ax.get_xlim()[1]+1, -4, 'Higher in non-fasted', ha='right', va='top', fontsize=5)
    sns.despine(left=True, ax=ax)
    
    
def lipid_coef_coef_plot(ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    df = data.loc[(data['Type'] == 'lipid') & (data['ID'] != 'Unidentified')].copy()
    unid = data.loc[(data['Type'] == 'metabolite') & (data['ID'] == 'Unidentified')].copy()

    outliers = {  
        'l_582': dict(x=0.05, y=0.0, name='TG 22:6_22:6_22:6'),  # 66:18
        'l_623': dict(x=0.03, y=-0.01, name='TG 62:13'),       # 62:13
        'l_716': dict(x=-0.0, y=-0.04, name='TG 60:10'),       # 60:10
        'l_569': dict(x=0.08, y=-0.3, name='TG 20:5_22:6_22:6'),  # 64:17
        'l_680': dict(x=0.15, y=-0.1, name='TG 18:0_20:5_22:6'),
        'l_644': dict(x=-0.05, y=0.11, name='TG 18:2_18:2_22:6'), # 58:10
        'l_910': dict(x=0.15, y=-0.26, name='TG 58:2'),        # 58:2
        'l_617': dict(x=0.3, y=-0.02, name='TG 18:1_20:5_20:5'),  # 58:11
        'l_888': dict(x=0.2, y=-0.2, name='TG 18:2_18:1_24:1'), # 50:4
        
        'l_372': dict(x=0.11, y=-0.03, name='PE 18:0_22:6'),  # PE 40:6
        'l_241': dict(x=-0.15, y=0.05, name='Plasmanyl-PC O-34:4'),
        'l_404': dict(x=-0.06, y=0.08, name='PE 18:0_20:4'),    # PE 38:4
        
        'l_10': dict(x=0.11, y=-0.20, name='AC 14:0'),
        'l_4': dict(x=0.15, y=-0.15, name='AC 18:2'),
        # 'l_1': dict(x=0.0, y=-0.25, name='AC 4:0'),
        
        # 'l_32': dict(x=-0.11, y=0.10, name='LysoPC 17:1'),
        'l_844': dict(x=-0.11, y=-0.24, name='CE 18:1'),
        # 'l_47': dict(x=-0.10, y=0.02, name='LysoPC 18:1'),
        'l_197': dict(x=-0.05, y=0.0, name='PC 38:7'),
        'l_507': dict(x=-0.2, y=0.1, name='SM d41:2'),
        'l_232': dict(x=-0.2, y=0.1, name='PC O-40:7'),
    }
    
    df['outlier'] = df.index.isin(outliers)
    df = df.sort_values('outlier')


    sns.scatterplot(
        data=df, x='coef_fed', y='coef_fasted', hue='superclass', ax=ax, palette=colors,
        size='outlier', sizes={True: BIG_SCATTER_SIZE, False: SMALL_SCATTER_SIZE}, 
        alpha=0.8, edgecolor='0.33', linewidth=0.33)
    ax.ticklabel_format(scilimits=(-1, 1))
    for i, row in data.loc[outliers].iterrows():
        x, y = row['coef_fed'], row['coef_fasted']        
        # https://stackoverflow.com/questions/29107800/python-matplotlib-convert-axis-data-coordinates-systems
        output_coords = ax.transLimits.transform((x, y))
        xtext = output_coords[0] + outliers[i]['x']
        ytext = output_coords[1] + outliers[i]['y']
        annot_text = row['lipid_class'] + ' ' + row['fa_carbon:unsat']
        ax.annotate(
            annot_text, xy=(x, y), xytext=(xtext, ytext), 
            textcoords='axes fraction',
            ha='center', annotation_clip=False, zorder=-5, va='bottom', color='0.5', fontsize=6,
            arrowprops=dict(width=0.6, headwidth=0.6, facecolor='gray', edgecolor='0.7'))
        ax.scatter(x, y, edgecolor='0.7', facecolor='white', linewidth=1, s=BIG_SCATTER_OUTLINE_SIZE, 
            zorder=-4)

    ax.set_ylabel('OGTT gluc. AUC regression slope\n(Fasted samples)', fontsize=6)
    ax.set_xlabel('OGTT gluc. AUC regression slope\n(Non-fasted samples)', fontsize=6)

    ax.axhline(0, color='0.05', linewidth=0.5, zorder=-99)
    ax.axvline(0, color='0.05', linewidth=0.5, zorder=-99)
    
    ax.yaxis.get_offset_text().set_fontsize(5)
    ax.xaxis.get_offset_text().set_fontsize(5)
    
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    
    ax.grid(linewidth=0.2, zorder=-200)
    ax.tick_params(length=0, pad=1, labelsize=6, labelcolor='0.05')
    handles, labels = ax.get_legend_handles_labels()
    handles = handles[1:-3]
    labels = labels[1:-3]
    
    legend = ax.legend(handles=handles, labels=labels, loc=(-0.41, 0.2), 
                       title='Molecule class', fontsize=6, title_fontsize=6,
                       markerscale=0.6, edgecolor='0.5', 
                       frameon=False, handletextpad=0.2, labelspacing=0.2, borderpad=0.1)
    sns.despine(left=True, bottom=True, ax=ax)
    return legend
        

def make_carbon_unsat_plot(
        lipid_class, jitter_offset, 
        base_size=50,
        ax=None, cax=None):
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4.5), dpi=150)
    df = add_jitter(lipid_class, ldata=data, os=jitter_offset)
    max_C, min_C = df['fa_carbons'].max(), df['fa_carbons'].min()
    max_unsat, min_unsat = df['fa_unsat'].max(), df['fa_unsat'].min()
    norm = plt.matplotlib.colors.CenteredNorm(vcenter=0.0,)  #  vmin=df['log2 FC'].min(), vmax=df['log2 FC'].max()
    cmap='coolwarm'
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    # display(df)
    sns.scatterplot(
        data=df, x='fa_carbons', y='fa_unsat', ax=ax, hue='Log2 Fold Change', hue_norm=norm, palette=cmap,  
        size='overlaps', sizes={1: base_size, 2: 0.75*base_size, 3: 0.6*base_size, 4: 0.4*base_size},
        # s=size, 
        legend=False, edgecolor='gray')
    ax.set_xticks(np.arange(min_C, max_C+1, 2))
    ax.set_xticklabels([int(x) for x in np.arange(min_C, max_C+1, 2)])
    ax.set_yticks(np.arange(min_unsat, max_unsat+1, 2))
    ax.set_yticklabels([int(x) for x in np.arange(min_unsat, max_unsat+1, 2)])
    ax.grid(color='#CCCCCC')
#     plt.legend(title='log2 fold change', loc=(1.01, 0.5), markerscale=1.5, fontsize=15, title_fontsize=18)
#     plt.title('Lipids by class significant under glucose tolerance', fontsize=18)
    ax.set_ylabel('Total fatty acyl unsaturations')
    ax.set_xlabel('Total fatty acyl carbons')
    
    cb = ax.figure.colorbar(sm, cax=cax, shrink=0.8, fraction=0.5, aspect=15, pad=0, 
                            label='\nlog2 fold change\n[Non-fasted – Fasted]')
    # cb.ax.tick_params(labelsize=8, length=0)
    cb.ax.set_yticks([-3, -2, -1, 0, 1, 2, 3], fontsize=5)
    shrink_cbar(cb.ax, shrink=0.7)
    # cb.ax.set_label('Log2 Fold Change')
    # cb.ax.set_title('Non-fasted\nHigher', fontsize=10, ha='center')
#     cb.ax.set_xticks(ticks=[0], labels=['Fasted\nHigher'], fontsize=20)
#     cb.ax.set_xticklabels(ticks=[0], labels=['Fasted\nHigher'], fontsize=20)
    sns.despine(ax=ax)
    return ax, cb

 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    