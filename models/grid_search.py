from sklearn.model_selection import GridSearchCV, 
import pandas as pd


def grid_search(X, y, model, params, cv, scoring, print_all=True, **kwargs):
    gs = GridSearchCV(estimator=model,
                      param_grid=params,
                      scoring=scoring,
                      cv=cv,
                      n_jobs=7,
                      error_score='raise',
                      **kwargs)
    result = gs.fit(X, y)
    print("Best: %f using %s" % (result.best_score_, result.best_params_))
    means = result.cv_results_['mean_test_score']
    stds = result.cv_results_['std_test_score']
    params = result.cv_results_['params']
    if print_all:
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
    l = []
    for mean, param in zip(means, params):
        d = {'mean': mean, **param}
        l.append(d)
    return pd.DataFrame(l)