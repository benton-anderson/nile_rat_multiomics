import numpy as np


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
    Input lipid string e.g. "TG 60:4" or "PC 18:2_20:1"
    Returns: tuple of (lipid class, SM label, # carbons, # unsat, sum composition string, FA list)
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
        fa = [[int(x) for x in sublist] for sublist in fa]
        carbons = sum([chain[0] for chain in fa])
        unsat = sum([chain[1] for chain in fa])
        sum_comp = str(carbons) + ':' + str(unsat)
        result = cls + ' ' + label + str(carbons) + ':' + str(unsat)
        return cls, label, carbons, unsat, sum_comp, fa
    except:
        # return a tuple of 6 nan
        return (np.nan,) * 6
        
        
def tight_bbox(ax):
    """
    Example for placing figure panel letters at very top left of the axis bounding box: 
    for ax, letter in zip([ax1, ax2], ['A', 'B']):
        bb = tight_bbox(ax)
        ax.text(x=bb.x0, y=bb.y1, s=letter, transform=fig.transFigure)
    """

    fig = ax.get_figure()
    tight_bbox_raw = ax.get_tightbbox(fig.canvas.get_renderer())
    from matplotlib.transforms import TransformedBbox
    tight_bbox_fig = TransformedBbox(tight_bbox_raw, fig.transFigure.inverted())
    return tight_bbox_fig
    