from sklearn.feature_selection import chi2
import numpy
import pandas as pd

def fisher_score_selection(train_set, fisher_score_threshold):
    x_train = train_set.iloc[:, :-1]
    y_train = train_set['label']
    f_score = chi2(x_train, y_train)
    scores = f_score[0]
    scores = scores[~numpy.isnan(scores)]

    indexes_above_thresh = pd.Series(scores).nlargest(fisher_score_threshold).index
    return indexes_above_thresh.tolist()
