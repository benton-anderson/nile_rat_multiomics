from sklearn.model_selection import GridSearchCV, RepeatedKFold, RepeatedStratifiedKFold
import pandas as pd
import math
from collections import defaultdict
import random


def custom_random_fasted_split10(X_all, n_repeats=5):
    """
    Only works for n_splits=10 on 30 total samples
    
    Because we are comparing only-random vs. only-fasted vs. all data, 
    only-random and only-fasted are at a disadvantage because they have 30 samples, 
    and 'all' has 60 samples, thus 'all' gets more training data.
    To compensate, at each split, pull random sample of 15 RBG and 15 FBG, 
    and get a list of (train, test) labels similar to the sklearn splitters.
    
    return list of (train_list, test_list) indexes
    """
    
    n_splits = 10
    
    sample_indices = pd.Series(range(len(X_all)), index=X_all.index)
    rbg_cols = X_all.filter(regex='RBG', axis=0).index
    fbg_cols = X_all.filter(regex='FBG', axis=0).index
    custom_cv = []  # custom_cv splitter must return a list of (train, test) tuples
    for i in range(n_repeats):
        random.seed(i)
        samples = sample_indices.loc[random.sample(rbg_cols.to_list(), 15)].to_list() + \
                  sample_indices.loc[random.sample(fbg_cols.to_list(), 15)].to_list()
        for j in range(n_splits):
            start = 3 * j  # start and end are very brittle because of the 3 * j. Therefore only use n_splits=10
            end = 3 * j + 3
            test = samples[start:end]
            train = list(set(samples).difference(test))
            custom_cv.append((train, test))
    return custom_cv
    

def data_subset_hyperparam_search(
    model, 
    param_grid, 
    scoring,
    X_all,  # 'y' parameter not needed because it appears in y_list
    columns_list, 
    column_names, 
    y_list, 
    y_names, 
    estimator_type,
    n_splits=10, 
    n_repeats=5, 
    n_jobs=-2, # -2 uses all but 1 core
    **kwargs
    ):
    """
    Performs grid search across data subsets (RBG vs. FBG vs. All),
    and y predictions (OGTT, Insulin, Weight, BG, etc.)
    
    Returns dataframe labeled with hyperparam grid search and prediction results 
    """
    results = defaultdict(dict)
    
    for column_name in column_names:
        if column_name not in ['RBG', 'FBG', 'all']:
            raise ValueError('Bad column names')
    
    for columns, column_name in zip(columns_list, column_names):
        for y, y_name in zip(y_list, y_names):
            if estimator_type == 'clf':
                cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1) 
            elif estimator_type == 'regress':
                if column_name == 'all':
                    cv = custom_random_fasted_split10(X_all=X_all, n_repeats=n_repeats)
                elif column_name == 'RBG' or column_name == 'FBG':
                    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)
                else: 
                    raise ValueError('column_name is not "all" or "RBG" or "FBG"')
            else:
                raise ValueError('cv must be one of "clf" or "regress"')
               
            print(column_name, y_name)
            
            gs = GridSearchCV(
                model, 
                param_grid,
                scoring=scoring,
                n_jobs=n_jobs,  
                cv=cv,
                verbose=2,  # verbosity prints to the Anaconda Command prompt, so intercept it? 
                **kwargs,)
            gs.fit(X_all.loc[columns], y.loc[columns])
            results[column_name][y_name] = {}
            results[column_name][y_name]['cv_results'] = gs.cv_results_
            results[column_name][y_name]['gs_obj'] = gs
    
    results = dict(results)
    dfs = []
    for columns in column_names:
        for y_name in y_names:
            df = pd.DataFrame(results[columns][y_name]['cv_results'])
            df['data_subset'] = columns
            df['y_type'] = y_name
            dfs.append(df)
    return pd.concat(dfs).reset_index(drop=True), results