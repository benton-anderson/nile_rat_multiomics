import numpy as np
import pandas as pd

data = pd.read_csv(r'../data/processed/combined_metabolites_data_with_model_params.csv').set_index('i')
data_cols = data.filter(regex='_FBG|_RBG').columns
fbg_cols = data.filter(regex='_FBG').columns
rbg_cols = data.filter(regex='_RBG').columns

ap = pd.read_excel(r'..\data\metadata\animal_phenotypes.xlsx', index_col=0)
fg = pd.read_csv(r'..\data\metadata\combined_metab_lipid_file_grouping.csv', index_col=0)


num_fas = {
    'FA': 1,
    'AC': 1,
    'LysoPC': 1,
    'LysoPE': 1,
    'LysoPI': 1,
    'CE': 1,
    'Plasmenyl-PE': 2, 
    'Plasmenyl-PC': 2,
    'Plasmanyl-PC': 2,
    'Plasmanyl-PE': 2,
    'PE': 2,
    'PC': 2,
    'PI': 2,
    'SM': 2,
    'DG': 2,
    'Alkenyl-DG': 2,
    'Cer[NS]': 2,
    'TG': 3,
}


def ppm_tol(a, b=None, tol=10):
    """
    Return tuple of (low m/z, high m/z) for one m/z value if b=None
    
    or Return boolean whether a and b are within tolerance of each other.
    """
    if b is None:
        mz_diff = a * tol * 1e-6
        return (a - mz_diff, a + mz_diff)
    elif b is not None:
        return abs(a - b) / a * 1e6 < tol


def parse_p_value(pval):
    if   pval < 0.0001:
        return '****'
    elif pval < 0.001:
        return '***'
    elif pval < 0.01:
        return '**'
    elif pval < 0.05:
        return '*'
    else:
        return 'ns'


def parse_lipid(lipid):
    """
    Input lipid string e.g. "TG 60:4", "PC 18:2_20:1", "SM d37:1"
    Returns: tuple of (
        lipid class, 
        class label, 
        # FA carbons, 
        # FA unsat, 
        sum composition string, 
        FA list, 
        is_sum_comp boolean
    )
    """
    try:
        l = lipid.split(' ')
        cls = l[0]
        fa = l[1]
        if '-' in fa:
            fa = fa.split('-')[1]
            label = fa.split('-')[0] + '-'
        else:
            label = ''
        if 'd' in fa:
            fa = fa.split('d')[1]
            label = fa.split('d')[0] + 'd'
        else:
            label = ''

        fa = [x.split(':') for x in fa.split('_')]
        fa = [tuple(int(x) for x in sublist) for sublist in fa]
        carbons = sum([chain[0] for chain in fa])
        unsat = sum([chain[1] for chain in fa])
        sum_comp = str(carbons) + ':' + str(unsat)
        result = cls + ' ' + label + str(carbons) + ':' + str(unsat)
        
        is_sum_comp = False
        if num_fas[cls] == 1:
            is_sum_comp = False
        elif len(fa) == 1:
            is_sum_comp = True
        
        return cls, label, carbons, unsat, sum_comp, fa, is_sum_comp
    except:
        # return a tuple of nan
        return (np.nan,) * 7
        
        
def tight_bbox(ax):
    """
    Example for placing figure panel letters at very top left of the axis bounding box: 
    for ax, letter in zip([ax1, ax2], ['A', 'B']):
        bb = tight_bbox(ax)
        ax.text(x=bb.x0, y=bb.y1, s=letter, transform=fig.transFigure, fontweight='bold')
    """
    fig = ax.get_figure()
    tight_bbox_raw = ax.get_tightbbox(fig.canvas.get_renderer())
    from matplotlib.transforms import TransformedBbox
    tight_bbox_fig = TransformedBbox(tight_bbox_raw, fig.transFigure.inverted())
    return tight_bbox_fig
    
    
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
    
    
def add_jitter(lipid_class, ldata, os=0.15):
    """
    For adjusting scatterplot points to avoid overlap in FA length vs. unsaturation plots. 
    """
    # os = offset for jitter
    tgdf = ldata.loc[data['molec_class'] == lipid_class].copy()
    tgdf['overlaps'] = 1
    overlaps = ldata['fa_carbon:unsat'].value_counts() > 1
    overlaps = overlaps.loc[overlaps == True].index
    # display(overlaps)
    for overlap in overlaps: 
        df = tgdf.loc[tgdf['fa_carbon:unsat'] == overlap]
        if len(df) == 4:
            for carbon_offset, unsat_offset, (i, row) in zip([-os,os,os,os], [os,-os,os,-os], df.iterrows()):
                tgdf.loc[i, 'fa_carbons'] = tgdf.loc[i, 'fa_carbons'] + carbon_offset
                tgdf.loc[i, 'fa_unsat'] = tgdf.loc[i, 'fa_unsat'] + unsat_offset
                tgdf.loc[i, 'overlaps'] = len(df)
        if len(df) == 3:
            for carbon_offset, unsat_offset, (i, row) in zip([0, -os, os], [os, -os, -os], df.iterrows()):
                tgdf.loc[i, 'fa_carbons'] = tgdf.loc[i, 'fa_carbons'] + carbon_offset
                tgdf.loc[i, 'fa_unsat'] = tgdf.loc[i, 'fa_unsat'] + unsat_offset
                tgdf.loc[i, 'overlaps'] = len(df)
        if len(df) == 2:
            for carbon_offset, unsat_offset, (i, row) in zip([-os, os], [0, 0], df.iterrows()):
                tgdf.loc[i, 'fa_carbons'] = tgdf.loc[i, 'fa_carbons'] + carbon_offset
                tgdf.loc[i, 'fa_unsat'] = tgdf.loc[i, 'fa_unsat'] + unsat_offset
                tgdf.loc[i, 'overlaps'] = len(df)
    return tgdf
    
    
def lipid_means(unsat_low, unsat_high, data):
    idx = data.loc[(data['fa_unsat'] >= unsat_low) & (data['fa_unsat'] <= unsat_high)].index
    df = (data
        .loc[idx, data_cols]
        .mean()
        .to_frame('quant')
        .join(fg[['ogtt', 'insulin', 'bg_type', 'week']])
        )
    return df
    
    
def format_number(num):
    """
    From https://stackoverflow.com/a/45846841/8659119
    """
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])