import numpy as np

def parse_lipid(lipid):
    """
    Input lipid string e.g. TG 60:4 or PC 18:2_20:1
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
    #     print(lipid, result)
        return cls, label, carbons, unsat, sum_comp, fa
    except:
        return (np.nan,) * 6
