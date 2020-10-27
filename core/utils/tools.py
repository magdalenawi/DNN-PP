from sklearn.model_selection import KFold, cross_val_score


def test_model_cv_nmae(model, x, y, cv=KFold(n_splits=5,shuffle=True)):
    scores = cross_val_score(model, x, y, cv=cv, n_jobs=-1, scoring='neg_mean_absolute_error')

    scores = -1*scores

    return scores.mean()


def test_model_cv_rmse(model, x, y, cv=KFold(n_splits=5,shuffle=True)):
    scores = cross_val_score(model, x, y, cv=cv, n_jobs=-1, scoring='neg_root_mean_squared_error')

    scores = -1*scores

    return scores.mean()


