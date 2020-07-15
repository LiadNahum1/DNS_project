import pandas as pd
from sklearn.svm import OneClassSVM
import pickle as pkl

from FeatureExtraction.name_time_features import build_features_for_chunk, TRAIN_PERCENT


def build_train_samples(user_id):
    features_of_user_id = pd.read_csv(f'../../FeatureExtraction/FeaturesPerUser/features_user_{user_id}.csv')
    train_size = round(len(features_of_user_id) * TRAIN_PERCENT)
    train_samples = features_of_user_id[0:train_size]  # only positive samples
    return train_samples

#user_id is the user that we check if the chunk belongs to him
def build_test_samples(user_id):
    with open('../FileCenter/all_user_chunks', 'rb') as fp:
        all_user_chunks = pkl.load(fp)
    test_samples = []
    for i in range(0, 2):
        user = all_user_chunks[i]
        train_size = round(len(user) * TRAIN_PERCENT)
        for chunk in user[train_size:]:
            test_samples.append(build_features_for_chunk(user_id, chunk))
    return test_samples


def main(user_id):
    train_samples = build_train_samples(user_id)
    test_samples = build_test_samples(user_id)
    clf = OneClassSVM(gamma='auto').fit(train_samples)
    print(pd.Series(clf.predict(test_samples)).value_counts())

if __name__ == "__main__":
    main(1)
