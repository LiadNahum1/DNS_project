from sklearn.feature_selection import chi2
import numpy
P_VALUE_THRESHOLD = 0.5


def fisher_score_selection(train_set):
    x_train = train_set.iloc[:, :-1]
    y_train = train_set['label']
    f_score = chi2(x_train, y_train)
    p_values = f_score[1]
    p_values = p_values[~numpy.isnan(p_values)]
    #print(x_train)
    indexes_above_thresh = numpy.where(p_values < P_VALUE_THRESHOLD)[0]
    #print(indexes_above_thresh)
    print(len(indexes_above_thresh))
    return indexes_above_thresh.tolist()