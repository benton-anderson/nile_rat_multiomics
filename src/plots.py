import json
from math import floor, ceil

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
import networkx as nx

from .utils import parse_p_value, parse_lipid, shrink_cbar

plt.style.use('../data/metadata/Nilerat_matplotlib_stylesheet.mplstyle')

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
data['log_qval_fed'] = -np.log10(data['qval_fed'])
data['log_qval_fasted'] = -np.log10(data['qval_fasted'])

short_names = {
        '4-Hydroxybutyric acid (GHB)': 'GHB',
        'alpha-Glycerylphosphorylcholine': 'GPC',
        '3-Hydroxybutyric acid': 'BHB',
        'Ethyl-beta-D-glucuronide': 'Ethyl gluc.',
        'Guanidinosuccinic acid': 'GSA',
        '4-Guanidinobutyric acid': '4-GBA',
        'N-Isovalerylglycine': 'IsovalGly',
        'Phenylacetylglycine': 'PhAcGly', 
        'Acrylic acid': 'Acrylate',
        '1,5-Anhydro-D-glucitol': 'Anhydrogluc.',
        'N-Methyl-2-pyrrolidone': 'Me-Pyrrolidone',    
        'N-Acetylneuraminic acid': 'NeuAc',
        '4-Hydroxybenzaldehyde': '4-OH-benzald.',
        
        'Phenylalanine': 'Phe', 'Asparagine': 'Asp', 'Alanine': 'Ala', 'Threonine': 'Thr', 
        'Glutamine': 'Gln', 'Leucine': 'Leu', 'Isoleucine': 'Ile', 'Glutamic acid': 'Glu', 
        'Histidine': 'His', 'Arginine': 'Arg', 'Tryptophan': 'Trp', 'Tyrosine': 'Tyr', 
        'Serine': 'Ser', 'Proline': 'Pro',}


def plot_quant_vs_ogtt(feature, x='ogtt', data=data,
                       xlabel=None, ylabel=None, 
                       animal_lines=False, legend=False,
                       robust=False, ax=None, scatter_kws=None, line_kws=None):
    """
    feature: 'm_123', or iterable of indexes ['m_123', 'l_123']
    x: 'ogtt' or 'insulin'
    """
    # 
    if ax is None:
        fig, ax = plt.subplots()
    df = data.loc[feature, data_cols]
    if not isinstance(feature, str):
        df = df.mean()
    df = (df
          .to_frame(name='quant')
          .join(fg[['bg_type', x]])
         )
        
    df['bg_type'] = df['bg_type'].replace('FBG', 'Fasted').replace('RBG', 'Non-fasted')
    df['quant'] = df['quant'].astype('float')
    for bg_type in df['bg_type'].unique():
        sns.regplot(data=df.loc[df['bg_type'] == bg_type], x=x, y='quant', n_boot=200, robust=robust, 
                    color=colors[bg_type], truncate=False, label=bg_type, scatter_kws=scatter_kws, line_kws=line_kws,
                    ax=ax, seed=1)
    # Can't use seaborn lmplot because it's a Figure-level plot, not Axes-level
    # sns.lmplot(data=df, x=x, y='quant', hue='bg_type', palette=colors, ax=ax,
    #            n_boot=200, truncate=False, scatter_kws=scatter_kws, line_kws=line_kws, seed=1)    
    if legend:
        ax.legend()
    if animal_lines:
        for unique_x in df[x].unique():
            ax.axvline(unique_x, color='gray', alpha=0.5, linewidth=0.5)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if x == 'ogtt':
        ax.set_xticks(ticks=[20000, 40000, 60000], labels=['20k', '40k', '60k'])
    sns.despine(ax=ax)
    return ax
    

def plot_graph(metab_set, corr=0.5, corr_type='spearman', 
               continuous_var='coef_fed', centered_norm=True, cmap='coolwarm',
               layout=nx.kamada_kawai_layout, use_connec_comp=True, 
               fontsize=5, pos_scale=1, max_linewidth=3.5,
               ax=None, cax=None):   
    """
    Make a connected graph with nodes and edges
    """
    if ax is None:
        fig, ax = plt.subplots(ncols=1, figsize=(5, 3), 
        # gridspec_kw=dict(width_ratios=(5, 1))
        )
        
    # Check if metab_set is list of 'm_100' or compound names
    if metab_set[0][1] != '_':
        list_length = len(metab_set)
        metab_set = data.loc[data['ID'].isin(metab_set)].index.to_list()
        if list_length != len(metab_set):
            raise ValueError('Could not find all metabolite names in metab_set')
    
    df = (data
     .loc[metab_set, rbg_cols]
     .T
     .corr(corr_type)
     .reset_index()
     .rename({'i': 'i1'}, axis=1)  # Prevents a bug
     .melt(id_vars='i1')
    )
    df.columns = ['to', 'from', 'corr']
    df = df.loc[(df['to'] != df['from']) & (df['corr'].abs() > corr)]
    g = nx.from_pandas_edgelist(df, 'to', 'from')
    
    if use_connec_comp:
        largest_group = max(list(nx.connected_components(g)), key=len)
        g = g.subgraph(largest_group)
        metab_set = list(g.nodes)
        
    # Get edgewidths for connecting lines
    edge_widths = []
    for edge in g.edges:
        corr = df.loc[(df['to'] == edge[0]) & (df['from'] == edge[1]), 'corr'].iloc[0]
        width = 0.05 + 0.4*1/-np.log(abs(corr) + 0.00001)  # fancy code to convert higher correlation value into thicker line
        if width > max_linewidth:  # Set a cap on linewidth 
            width = max_linewidth
        edge_widths.append(width)
        
    # Colorbar
    continuous_values = data.loc[metab_set, continuous_var].to_list()
    if centered_norm:
        norm = plt.matplotlib.colors.CenteredNorm()
        norm(continuous_values)
    else: 
        norm = plt.matplotlib.colors.Normalize(vmin=min(continuous_values), vmax=max(continuous_values))
    sm = plt.matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(continuous_values)
    
    pos = layout(g)
    for key, val in pos.items():
        pos[key] = val * pos_scale
    
    for i, (x, y) in pos.items():
        name = data.loc[i, 'ID']
        if name in short_names:
            name = short_names[name]
        cont_value = data.loc[i, continuous_var]
        c = sm.to_rgba(cont_value)[:-1]
        alpha = 0.2
        lighter_c = [x + (1 - x) * (1 - alpha) for x in c]
        bbox_style = dict(boxstyle='round4', pad=.3, mutation_scale=0, linewidth=0.7, 
                          edgecolor=c, facecolor=lighter_c, alpha=1)
        text = ax.text(x, y, name.replace(' ', '\n'), 
                       ha='center', va='center', bbox=bbox_style, fontsize=fontsize, fontweight='semibold')
    nx.draw_networkx_edges(g, pos=pos, ax=ax, width=edge_widths, 
                           edge_color='0.33',  # edge_color=[(pos_corr_color if x else neg_corr_color) for x in is_pos_corr]
                          )
    cbar = ax.get_figure().colorbar(mappable=sm, cax=cax)
    cbar.outline.set_linewidth(0.3)
    cax = cbar.ax
    shrink_cbar(ax=cax, shrink=0.7)
    cax.yaxis.set_label_position('right')
    cax.yaxis.tick_right()
    cax.tick_params(length=2, pad=1, labelsize=fontsize)
    cax.set_ylabel('OGTT gluc. AUC \nregression slope', # $mg\cdot min \cdot dL^{-1}$
                   labelpad=6, rotation=90, ha='center', va='center', fontsize=fontsize)  
    cax.yaxis.get_offset_text().set_fontsize(fontsize)
    
    handles = [
    #     patches.Patch(color=pos_corr_color, label='Positive profile corr.'),
    #     patches.Patch(color=neg_corr_color, label='Negative profile corr.'),
        plt.matplotlib.patches.Patch(color='0.2', label='Low profile\ncorrelation'),
        plt.matplotlib.patches.Patch(color='0.2', label='High profile\ncorrelation')]
    # legend = ax1.legend(handles=handles, loc=(-0.1, 1), fontsize=fontsize)
    # for patch, height in zip(legend.legendHandles, [2, 8]):
    #     patch.set_width(15)
    #     patch.set_height(height)
    sns.despine(left=True, bottom=True, ax=ax)
    return ax, cbar

# DEFINE CONSTANTS FOR THE VOLCANO AND SLOPE_VS_SLOPE PLOTS
ANNOT_CIRCLE_SIZE = 90
ANNOT_COLOR = '0.5'
ANNOT_FONTSIZE = 6
ANNOT_LW = 1
POINT_SIZES = {True: 46, False: 22}
POINT_LW = 0.4    # LW = linewidth
POINT_EC = '0.1'  # EC = edgecolor
POINT_ALPHA = 1.0
SPINE_LW = 1
SPINE_GRID_COLOR = '0.1'
TICK_FONTSIZE = 6
TICK_PAD = 1
LABEL_FONTSIZE = 7

def _scatter(x, y, df, metab_type, 
             size, sizes, alpha, 
             ax, show_legend, plot_unid):
    """
    Base scatter plotting for volcano and slope_vs_slope plots. 
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,4), dpi=100)
    df = df.sort_values(size)  # sort by the size factor to ensure annotated points appear on top 
    sns.scatterplot(
        data=df.loc[(df['superclass'] != 'Unidentified') & (df['Type'] == metab_type)], 
        x=x, y=y, hue='superclass', palette=colors, 
#         s=30, linewidth=0.2, edgecolor='gray',
        size=size, sizes=sizes,
        edgecolor=POINT_EC, linewidth=POINT_LW,
        ax=ax, alpha=alpha, legend=show_legend,
        zorder=3)
    if plot_unid:
        sns.scatterplot(
            data=df.loc[(df['superclass'] == 'Unidentified') & (df['Type'] == metab_type)], 
            x=x, y=y, hue='superclass', palette=colors, ax=ax, 
            size=size, sizes=sizes,
            alpha=0.3, legend=show_legend,
            zorder=2)
    return ax

def volcano(x, y, df, metab_type, size, sizes=POINT_SIZES, alpha=POINT_ALPHA, ax=None, show_legend=False, plot_unid=False):
    """
    
    """
    _scatter(x=x, y=y, df=df, metab_type=metab_type, 
             size=size, sizes=sizes, alpha=alpha, ax=ax, show_legend=show_legend, plot_unid=plot_unid)
    # Draw the y-axis in the middle of the plot
    ax.axvline(0, linewidth=SPINE_LW, c=SPINE_GRID_COLOR, zorder=-99)
    ax.set_yticks([])
    ax.set_ylabel(None)
    ax.text(s='-Log10(q-value)', x=-0.05, y=ax.get_ylim()[1], ha='right', va='top', 
            rotation=90, color=SPINE_GRID_COLOR, fontsize=LABEL_FONTSIZE)
    for tick in range(0, round(ax.get_ylim()[1]), 5):
        ax.text(x=0.05, y=tick, s=tick, ha='left', va='center', 
                zorder=-20, color=SPINE_GRID_COLOR, fontsize=TICK_FONTSIZE)
    
    
    
    # x-axis ticks and label 
    ax.tick_params(axis='x', length=2, pad=TICK_PAD, labelsize=TICK_FONTSIZE)
    xlim = ax.get_xlim()
    ax.set_xlim(floor(xlim[0]), ceil(xlim[1]))
    xticks = range(floor(xlim[0]), ceil(xlim[1]) + 1, 1)
    ax.set_xticks(xticks)
    ax.set_xlabel('Log2(fold change) [Nonfasted - fasted]', fontsize=LABEL_FONTSIZE)
    
    # Write helper text 
    ax.text(ax.get_xlim()[0]-0.3, -1, 'Fasted more abundant', ha='left', fontsize=5)
    ax.text(ax.get_xlim()[1], -1, 'Non-fasted more abundant', ha='right', fontsize=5)
    
    if show_legend:
        ax.legend(loc=(1.01, 0.1), markerscale=1.2)
    
    ax.axhline(-np.log10(0.05), linewidth=SPINE_LW, c=SPINE_GRID_COLOR, zorder=-99)
    sns.despine(ax=ax, left=True)
    
    # Equal aspect ratio is important to not bias the viewer
    ax.set_aspect('equal')
    
    return ax


def slope_vs_slope(df, x, y, metab_type, size, sizes=POINT_SIZES, alpha=POINT_ALPHA, ax=None, show_legend=False, plot_unid=False):
    _scatter(x=x, y=y, df=df, metab_type=metab_type, 
             size=size, sizes=sizes, alpha=alpha, ax=ax, show_legend=show_legend, plot_unid=plot_unid)
    ax.set_xlabel('OGTT slope Non-fasted', fontsize=LABEL_FONTSIZE)    
    ax.set_ylabel('OGTT slope Fasted', fontsize=LABEL_FONTSIZE)
    ax.tick_params(length=0, pad=TICK_PAD, labelsize=TICK_FONTSIZE)
    
    # Set ticks every 2e-5
    ax.ticklabel_format(style='plain')  # style='plain' allows for re-labeling of ticks
    ticks = np.arange(-20, 20, 2)
    ax.set_xticks(ticks*1e-5, ticks)
    ax.set_yticks(ticks*1e-5, ticks)
    
    # Custom grid and spines on x=0 and y=0
    ax.grid(linewidth=0.2, color=SPINE_GRID_COLOR)
    ax.set_axisbelow(True)
    ax.axhline(0, color=SPINE_GRID_COLOR, lw=SPINE_LW, zorder=-99)
    ax.axvline(0, color=SPINE_GRID_COLOR, lw=SPINE_LW, zorder=-99)
    
    # Equal aspect ratio is important to not bias the viewer
    ax.set_aspect('equal')
    
    sns.despine(ax=ax, left=True, bottom=True)         
    return ax


def annotate_point(xy,  
                   text, 
                   xytext, 
                   ax,
                   relpos=(0.5, 0.5),
                   lw=ANNOT_LW,
                   color=ANNOT_COLOR,
                   fontsize=ANNOT_FONTSIZE,
                   ha='center',
                   zorder=-20,
                   circle_size=ANNOT_CIRCLE_SIZE
                  ):
    """
    Convenience function for streamlining annotation 
    on slope vs. slope and volcano plots. 
    
    Position of text is given in the difference in 'axes fraction' 
    away from the point for easier eyeballing. 
    
    xytext values should be a percentage (e.g. 8) rather than a decimal (0.08) 
    
    """
    frac_to_data = ax.transLimits.inverted().transform  # converts axes fraction to data coords
    # To calculate the xytext as a delta, find the difference in data coords from (0, 0)
    xtextdelta, ytextdelta = (frac_to_data(xytext) - frac_to_data([0, 0])) / 100
    xytext = (xy[0] + xtextdelta, xy[1] + ytextdelta)
    ax.annotate(
        text=text, 
        xy=xy, 
        xytext=xytext, #(counter+5, row['coef']+0.05),
        textcoords='data',
        arrowprops=dict(arrowstyle='-', 
                        relpos=relpos, 
                        lw=lw, 
                        color=color),
        bbox=dict(pad=0, 
                  facecolor='white', 
                  edgecolor='none'),
        fontsize=fontsize, 
        annotation_clip=True, 
        ha=ha, va='center',
        zorder=zorder,
    )
    # Draw circle around point
    ax.scatter(xy[0], xy[1], edgecolor=color, facecolor='white', linewidth=lw, 
               s=circle_size, zorder=zorder)
   
    
def set_square_ratio(ax):
    """
    Adjust plot to have square ratio when axes have different limits.
    This minimizes bias against the axis with larger range. 
    Used in the linear regression slope vs. slope scatter plots. 
    
    ax.set_aspect('equal') is not ideal because it distorts the plot
    """
    # Get larger of two axis limits ranges
    xlim_range = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
    ylim_range = abs(ax.get_ylim()[1] - ax.get_ylim()[0])
    max_range = max(xlim_range, ylim_range)
    if xlim_range > ylim_range:
        small_range_get = ax.get_ylim
        small_range_set = ax.set_ylim
    else:
        small_range_get = ax.get_xlim
        small_range_set = ax.set_xlim
    # Get proportion of smaller axis limit that is greater than zero.
    #    this preserves the centering of the plot and ensures all points
    #    remain inside visible plotting area. 
    proportion = small_range_get()[1] / abs(small_range_get()[0] - small_range_get()[1])
    # set new limits on smaller axis
    small_range_set(-1*((1-proportion) * max_range), (proportion * max_range))


def fasted_fed_slope_old(_type, ax=None, alpha=0.8, legend=False):
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