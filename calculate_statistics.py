import pickle
import pandas as pd
import numpy


def get_false_positive_rate(test_set, predicted):
    real_positive = test_set.loc[test_set['label'] == 1].index.tolist()
    real_negative = len(test_set) - len(real_positive)
    predict_positive_indexes = numpy.where(predicted == 1)[0]
    is_true_positive_arr = numpy.isin(predict_positive_indexes, real_positive)
    false_positive = len(numpy.where(~is_true_positive_arr)[0])
    return false_positive / real_negative


def get_false_negative_rate(test_set, predicted):
    real_negative = test_set.loc[test_set['label'] == 0].index.tolist()
    real_positive = len(test_set) - len(real_negative)
    predict_negative_indexes = numpy.where(predicted == 0)[0]
    is_true_negative_arr = numpy.isin(predict_negative_indexes, real_negative)
    false_negative = len(numpy.where(~is_true_negative_arr)[0])
    return false_negative / real_positive


def main(user_id):
    with open('FileCenter/predicted', 'rb') as fp:
        predicted = pickle.load(fp)
    test_set = pd.read_csv(f'FileCenter/FeaturesPerUser/user{user_id}_test_features.csv')
    print(f'false positive rate {get_false_positive_rate(test_set, predicted)}')
    print(f'false negative rate {get_false_negative_rate(test_set, predicted)}')


if __name__ == '__main__':
    main(0)
