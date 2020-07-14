import pandas as pd
from sklearn.svm import OneClassSVM

TRAIN_PERCENT = 0.9
TEST_PERCENT = 0.1
USER_COUNT = 15


def build_train_samples(user_id):
    features_of_user_id = pd.read_csv(f'features_user_{user_id}.csv')
    train_size = round(len(features_of_user_id) * TRAIN_PERCENT)
    train_samples = features_of_user_id[0:train_size]  # only positive samples
    return train_samples


def build_test_samples(user_id, other_user_id):
    features_of_user_id = pd.read_csv(f'features_user_{user_id}.csv')
    train_size = round(len(features_of_user_id) * TRAIN_PERCENT)
    test_samples = features_of_user_id[train_size:]
    return test_samples


def main(user_id, other_user_id):
    train_samples = build_train_samples(user_id)
    test_samples = build_test_samples(user_id, other_user_id)
    clf = OneClassSVM(gamma='auto').fit(train_samples)
    print(clf.predict(test_samples))
