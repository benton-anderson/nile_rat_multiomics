from statsmodels.api import formula as smf
from statsmodels.stats.multitest import fdrcorrection
import pandas as pd
import matplotlib.pyplot as plt

ap = pd.read_excel(r'..\data\metadata\animal_phenotypes.xlsx', index_col=0)
fg = pd.read_csv(r'..\data\metadata\combined_metab_lipid_file_grouping.csv', index_col=0)

def quant_vs_ogtt_stats(data, features, data_cols, data_name, show_hist=True):
    """
    Performs ordinary linear regression on all features ain a dataframe. 
    Returns slope coefficients, R2, and FDR-corrected p-values for slopes 
    
    Observations (plasma samples) on rows, features on columns.
    
    data_name is 'fed' or 'fasted' or 'RBG' or similar
    """
    df = data.loc[features, data_cols].copy()
    df['ogtt'] = df.join(fg[['ogtt', 'sex', 'animal', 'cohort']])
    df[[f'pval_{data_name}', f'qval_{data_name}', f'coef_{data_name}']] = None
    
    d = {}
    for feature in features:
        model = smf.ols(f'{feature} ~ ogtt', data=df).fit()
        d[feature] = {'coef': model.
        
    if show_hist:
        plt.hist(df[f'pval_{data_name}'], bins=20)
        plt.title('p-values')
        plt.figure()
        plt.hist(df[f'qval_{data_name}'], bins=20)
        plt.title('FDR corrected q-values')
        plt.figure()
        plt.hist(df[f'coef_{data_name}'], bins=20)
        plt.title('slope coefficients')


























