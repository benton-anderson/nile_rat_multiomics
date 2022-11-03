import json
from math import floor, ceil

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
import networkx as nx

from .utils import parse_p_value, parse_lipid, shrink_cbar, add_jitter

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
FIG_LETTER_FONTSIZE = 9
FIG_LETTER_FONTWEIGHT = 'bold'
LEGEND_TEXT_FONTSIZE = 6
LEGEND_TITLE_FONTSIZE = 7
GRID_LW = 0.6
GRID_LIGHT_COLOR = '0.8'
HIGHLIGHT_BBOX_PAD = 1.5
HIGHLIGHT_FONTSIZE = 6
HIGHLIGHT_FONTWEIGHT = 'bold'
HIGHLIGHT_FACECOLOR = '0.9'
HIGHLIGHT_ANNOT_LW = 1.5

def _scatter(x, y, df, metab_type, 
             size, sizes, alpha, 
             ax, show_legend, plot_unid):
    """
    Base scatter plotting for volcano and slope_vs_slope plots. 
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,4), dpi=100)
    df = df.sort_values(size)  # sort by the size factor to ensure annotated points appear on top 
    # First plot the annotated 'outlier' points with highest zorder
    sns.scatterplot(
        data=df.loc[(df['superclass'] != 'Unidentified') & (df['Type'] == metab_type) & (df[size])], 
        x=x, y=y, hue='superclass', palette=colors, 
#         s=30, linewidth=0.2, edgecolor='gray',
        size=size, sizes=sizes,
        edgecolor=POINT_EC, linewidth=POINT_LW,
        ax=ax, alpha=alpha, legend=False,
        zorder=10)
    # Next plot the non-annotated points with a lower zorder
    sns.scatterplot(
        data=df.loc[(df['superclass'] != 'Unidentified') & (df['Type'] == metab_type) &(~df[size])], 
        x=x, y=y, hue='superclass', palette=colors, 
#         s=30, linewidth=0.2, edgecolor='gray',
        size=size, sizes=sizes,
        edgecolor=POINT_EC, linewidth=POINT_LW,
        ax=ax, alpha=alpha, legend=show_legend,
        zorder=5)
    if plot_unid:
        sns.scatterplot(
            data=df.loc[(df['superclass'] == 'Unidentified') & (df['Type'] == metab_type)], 
            x=x, y=y, hue='superclass', palette=colors, ax=ax, 
            size=size, sizes=sizes,
            alpha=0.3, legend=show_legend,
            zorder=1)
    return ax

def volcano(x, y, df, metab_type, size, sizes=POINT_SIZES, 
            alpha=POINT_ALPHA, xlim_override=None, ax=None, 
            show_legend=False, plot_unid=False):
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
    ax.set_xlim(floor(xlim[0])-0.3, ceil(xlim[1])+0.2)
    xticks = range(floor(xlim[0]), ceil(xlim[1]) + 1, 1)
    ax.set_xticks(xticks)
    ax.set_xlabel('Log2(fold change) [Nonfasted - fasted]', fontsize=LABEL_FONTSIZE)
    
    if xlim_override is not None:
        ax.set_xlim(xlim_override)
    
    # Write helper text 
    ax.text(ax.get_xlim()[0], -1, 'Fasted more abundant', ha='left', fontsize=5)
    ax.text(ax.get_xlim()[1], -1, 'Non-fasted more abundant', ha='right', fontsize=5)
    
    if show_legend:
        ax.legend(loc=(1.01, 0.1), markerscale=1.2)
    
    ax.axhline(-np.log10(0.05), linewidth=SPINE_LW, c=SPINE_GRID_COLOR, zorder=-99)
    sns.despine(ax=ax, left=True)
    
    return ax


def slope_vs_slope(df, x, y, metab_type, size, sizes=POINT_SIZES, alpha=POINT_ALPHA, ax=None, 
show_legend=False, plot_unid=False, aspect_equal=True):
    _scatter(x=x, y=y, df=df, metab_type=metab_type, 
             size=size, sizes=sizes, alpha=alpha, ax=ax, show_legend=show_legend, plot_unid=plot_unid)
    ax.set_xlabel('OGTT slope Non-fasted', fontsize=LABEL_FONTSIZE)    
    ax.set_ylabel('OGTT slope Fasted', fontsize=LABEL_FONTSIZE)
    ax.tick_params(length=0, pad=TICK_PAD, labelsize=TICK_FONTSIZE)
    
    # Set ticks every 2e-5
    ax.ticklabel_format(style='plain')  # style='plain' allows for re-labeling of ticks
    ticks = np.arange(-30, 30, 2)
    ax.set_xticks(ticks*1e-5, ticks)
    ax.set_yticks(ticks*1e-5, ticks)
    
    # Custom grid and spines on x=0 and y=0
    ax.grid(linewidth=GRID_LW, color=GRID_LIGHT_COLOR)
    ax.set_axisbelow(True)
    ax.axhline(0, color=SPINE_GRID_COLOR, lw=SPINE_LW, zorder=1)
    ax.axvline(0, color=SPINE_GRID_COLOR, lw=SPINE_LW, zorder=1)
    
    # Equal aspect ratio is important to not bias the viewer
    if aspect_equal:
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
                   fontweight='regular',
                   ha='center',
                   zorder=6,
                   circle_size=ANNOT_CIRCLE_SIZE, 
                   bbox_pad=0,
                   bbox_facecolor='white',
                   highlight=False,
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
    if highlight:
        bbox_pad = HIGHLIGHT_BBOX_PAD
        fontsize = HIGHLIGHT_FONTSIZE
        fontweight = HIGHLIGHT_FONTWEIGHT
        bbox_facecolor = HIGHLIGHT_FACECOLOR
        lw = HIGHLIGHT_ANNOT_LW
        zorder = zorder + 1
    ax.annotate(
        text=text, 
        xy=xy, 
        xytext=xytext, #(counter+5, row['coef']+0.05),
        textcoords='data',
        arrowprops=dict(arrowstyle='-', 
                        relpos=relpos, 
                        lw=lw, 
                        color=color),
        bbox=dict(pad=bbox_pad, 
                  facecolor=bbox_facecolor, 
                  edgecolor='none'),
        fontsize=fontsize, 
        fontweight=fontweight,
        annotation_clip=True, 
        ha=ha, va='center',
        zorder=zorder,
    )
    # Draw circle around point
    alpha_corrected_color = (str(float(color) - 0.15))
    ax.scatter(xy[0], xy[1], 
               edgecolor=alpha_corrected_color, facecolor='white', linewidth=lw, alpha=0.7,
               s=circle_size, zorder=zorder)


def plot_quant_vs_ogtt(feature, x='ogtt', data=data,
                       xlabel=None, ylabel=None, 
                       animal_lines=False, 
                       legend=False, legend_loc=(0.85, 0.02),
                       robust=False, ax=None, 
                       palette=colors,
                       alpha=1, 
                       scatter_kws=None, 
                       line_kws=dict(lw=1.5)):
    """
    feature: 'm_123', or iterable of indexes ['m_123', 'l_123']
    x: 'ogtt' or 'insulin'
    Font sizes and design have been optimized for figsize=(2.5, 2)
    
    """
    # 
    if ax is None:
        fig, ax = plt.subplots(dpi=250, figsize=(2.5, 2))
    df = data.loc[feature, data_cols]
    if not isinstance(feature, str):
        df = df.mean()
    df = (df
          .to_frame(name='quant')
          .join(fg[['bg_type', x]])
         )
    # Automatically include lw=0 to avoid the wrong-alpha edge color on scatter points
    if scatter_kws is None:
        scatter_kws = dict(lw=0)
    elif 'lw' not in scatter_kws:
        scatter_kws['lw'] = 0
    scatter_kws['alpha'] = alpha
    
    df['bg_type'] = df['bg_type'].replace('FBG', 'Fasted').replace('RBG', 'Non-fasted')
    df['quant'] = df['quant'].astype('float')
    for bg_type in df['bg_type'].unique():
        sns.regplot(data=df.loc[df['bg_type'] == bg_type], x=x, y='quant', n_boot=200, robust=robust, 
                    color=palette[bg_type], 
                    truncate=False, label=bg_type, 
                    scatter_kws=scatter_kws, line_kws=line_kws,
                    ax=ax, seed=1)
    # Can't use seaborn lmplot because it's a Figure-level plot, not Axes-level
    # sns.lmplot(data=df, x=x, y='quant', hue='bg_type', palette=colors, ax=ax,
    #            n_boot=200, truncate=False, scatter_kws=scatter_kws, line_kws=line_kws, seed=1)    
    ax.tick_params(pad=TICK_PAD, length=2, labelsize=TICK_FONTSIZE)
    if x == 'ogtt':
        ax.set_xticks(ticks=[20000, 40000, 60000], labels=['20k', '40k', '60k'])
    ax.set_xlabel('OGTT glucose AUC', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Log2 abundance', fontsize=LABEL_FONTSIZE)
    
    if legend:
        legend = ax.legend(
            loc=legend_loc,     
            title='Sampling', fontsize=LEGEND_TEXT_FONTSIZE, 
            title_fontproperties=dict(size=LEGEND_TITLE_FONTSIZE, weight='bold'),
            handletextpad=-0.1, labelspacing=0.15, borderaxespad=0, handlelength=1.7,
            borderpad=0.1, markerscale=1.,
#           frameon=True, framealpha=1, facecolor='0.95', fancybox=False, edgecolor='none'
        )
        legend._legend_box.align = 'left'  # shift legend title to left alignment
    if animal_lines:
        for unique_x in df[x].unique():
            ax.axvline(unique_x, color='gray', alpha=0.5, linewidth=0.5)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    
    sns.despine(ax=ax)
    return ax
 

def carbon_unsat(
        lipid_class, jitter_offset, 
        base_size=28,
        ax=None, cax=None):
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4.5), dpi=150)
    df = add_jitter(lipid_class, ldata=data, os=jitter_offset)
    max_C, min_C = df['fa_carbons'].max(), df['fa_carbons'].min()
    max_unsat, min_unsat = df['fa_unsat'].max(), df['fa_unsat'].min()
    norm = plt.matplotlib.colors.CenteredNorm(vcenter=0.0,)  #  vmin=df['log2 FC'].min(), vmax=df['log2 FC'].max()
    cmap = 'coolwarm'
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    # display(df)
    sns.scatterplot(
        data=df, x='fa_carbons', y='fa_unsat', ax=ax, hue='Log2 Fold Change', hue_norm=norm, palette=cmap,  
        size='overlaps', sizes={1: base_size, 2: 0.75*base_size, 3: 0.6*base_size, 4: 0.4*base_size},
        # s=size, 
        legend=False, edgecolor=POINT_EC, linewidth=0.3)
    ax.tick_params(pad=TICK_PAD, labelsize=TICK_FONTSIZE)
    ax.set_xticks(np.arange(min_C, max_C+1, 2), [int(x) for x in np.arange(min_C, max_C+1, 2)])
    ax.set_yticks(np.arange(min_unsat, max_unsat+1, 2), [int(x) for x in np.arange(min_unsat, max_unsat+1, 2)])
    ax.grid(color=SPINE_GRID_COLOR, lw=GRID_LW)
#     plt.legend(title='log2 fold change', loc=(1.01, 0.5), markerscale=1.5, fontsize=15, title_fontsize=18)
#     plt.title('Lipids by class significant under glucose tolerance', fontsize=18)
    ax.set_ylabel(f'{lipid_class} fatty acyl unsaturations', fontsize=LABEL_FONTSIZE)
    ax.set_xlabel(f'{lipid_class} fatty acyl carbons', fontsize=LABEL_FONTSIZE)
    
    cb = ax.figure.colorbar(sm, cax=cax, shrink=0.8, fraction=0.5, aspect=15, pad=0)
    # cb.ax.tick_params(labelsize=8, length=0)
    cb.ax.set_yticks([-3, -2, -1, 0, 1, 2, 3], fontsize=5)
    cb.ax.set_ylabel('Log2 fold change\n[Nonfasted – Fasted]', fontsize=LABEL_FONTSIZE)
    cb.ax.yaxis.set_label_position('left')
    cb.ax.tick_params(pad=TICK_PAD, labelsize=TICK_FONTSIZE)
    shrink_cbar(cb.ax, shrink=0.7)
    # cb.ax.set_label('Log2 Fold Change')
    # cb.ax.set_title('Non-fasted\nHigher', fontsize=10, ha='center')
#     cb.ax.set_xticks(ticks=[0], labels=['Fasted\nHigher'], fontsize=20)
#     cb.ax.set_xticklabels(ticks=[0], labels=['Fasted\nHigher'], fontsize=20)
    sns.despine(ax=ax)
    return ax, cb


def plot_network(metab_set, corr=0.5, corr_type='spearman', 
                 continuous_var='coef_fed', centered_norm=True, cmap='coolwarm',
                 layout=nx.kamada_kawai_layout, use_connec_comp=True, 
                 fontsize=6, pos_scale=1, 
                 max_linewidth=3.5, min_linewidth=0.2,
                 shrink_cbar_factor=0.6,
                 node_lw=0.7,
                 lightness_alpha=0.2,
                 ax=None, cax=None):   
    """
    Make a connected graph with nodes and edges.
    
    use_connec_comp means it will only pick the biggest subgraph that is connected.
        Not all nodes will be shown with this param.
    
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
        width = min_linewidth + 0.4*1/-np.log(abs(corr) + 0.00001)  # convert higher correlation value into thicker line
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
        alpha = lightness_alpha
        lighter_c = [x + (1 - x) * (1 - alpha) for x in c]  # make a lighter color without using alpha
        bbox_style = dict(boxstyle='round4', pad=.3, mutation_scale=0, linewidth=node_lw, 
                          edgecolor=c, facecolor=lighter_c, alpha=1)
        text = ax.text(x, y, name.replace(' ', '\n'), 
                       ha='center', va='center', bbox=bbox_style, fontsize=fontsize, fontweight='semibold')
    nx.draw_networkx_edges(g, pos=pos, ax=ax, width=edge_widths, 
                           edge_color='0.33',  # edge_color=[(pos_corr_color if x else neg_corr_color) for x in is_pos_corr]
                          )
    cbar = ax.get_figure().colorbar(mappable=sm, cax=cax)
    cbar.outline.set_linewidth(0.3)
    cax = cbar.ax
    shrink_cbar(ax=cax, shrink=shrink_cbar_factor)
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
    return ax, cbar, g


def custom_legend(entries, ax, loc=(1.02, 0), show_frame=False, sort=True, 
                  mew=POINT_LW, mec=POINT_EC, ms=8, marker='o', **kwargs):
    """
    Wrapper for making a legend based on list of entries, using colors defined in the main color scheme.
    """
    entries = sorted(entries)
    handles = []
    for entry in entries:
        color = colors[entry]
        handles.append(
            plt.matplotlib.lines.Line2D(
                [0], [0], label=entry,
                linewidth=0, mfc=color, mew=mew, mec=mec, ms=ms, marker=marker,
            )
        )
    if show_frame:
        frame_params = dict(frameon=True, framealpha=1, facecolor='0.95', fancybox=False, edgecolor='none')
    else:
        frame_params = dict()
    legend = ax.legend(
        handles=handles, loc=loc, 
        title_fontproperties=dict(size=LEGEND_TITLE_FONTSIZE, weight='bold'), 
        **frame_params,
        **kwargs
        )
    legend._legend_box.align = 'left'  # shift legend title to left alignment
    return legend


def shrink_cbar(ax, shrink=0.9):
    """
    Shrink the height of a colorbar that is set within its own pre-made colorbar axes (cax) 
    From https://stackoverflow.com/questions/53429258/matplotlib-change-colorbar-height-within-own-axes    
    """
    b = ax.get_position()
    new_h = b.height*shrink
    pad = (b.height-new_h)/2.
    new_y0 = b.y0 + pad
    new_y1 = b.y1 - pad
    b.y0 = new_y0
    b.y1 = new_y1
    ax.set_position(b)

    
def tight_bbox(ax):
    """
    Example for placing figure panel letters at very top left of the axis bounding box: 
    for ax, letter in zip([ax1, ax2], ['A', 'B']):
        bb = tight_bbox(ax)
        ax.text(x=bb.x0, y=bb.y1, s=letter, 
        fontsize=src.plots.LABEL_FONTSIZE, fontweight='bold', transform=fig.transFigure, )
    """
    fig = ax.get_figure()
    tight_bbox_raw = ax.get_tightbbox(fig.canvas.get_renderer())
    from matplotlib.transforms import TransformedBbox
    tight_bbox_fig = TransformedBbox(tight_bbox_raw, fig.transFigure.inverted())
    return tight_bbox_fig
    


def change_width(ax, new_value):
    """
    Offsetting boxplot and swarmplot side-by-side is super annoying.
    code from: 
    # https://stackoverflow.com/questions/61647192/boxplot-and-data-points-side-by-side-in-one-plot  
    
    """
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - new_value

        # change patch width
        patch.set_width(new_value)

        # re-center patch
        patch.set_x(patch.get_x() + diff * .5)


   
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