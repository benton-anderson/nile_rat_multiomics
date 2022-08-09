import numpy as np


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