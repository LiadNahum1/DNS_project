import pickle

import pandas as pd
from sklearn.svm import OneClassSVM
import pickle as pkl

from FeatureExtraction.name_time_features import build_features_for_chunk, TRAIN_PERCENT


def build_train_samples(user_id):
    with open('all_user_chunks', 'rb') as fp:
        all_user_chunks = pkl.load(fp)
    train_samples = []
    for i in range(0, len(all_user_chunks)):
        user = all_user_chunks[i]
        train_size = round(len(user) * TRAIN_PERCENT)
        for chunk in user[:train_size]:
            train_samples.append(build_features_for_chunk(user_id, chunk))
    return train_samples

#user_id is the user that we check if the chunk belongs to him
def build_test_samples(user_id):
    with open('all_user_chunks', 'rb') as fp:
        all_user_chunks = pkl.load(fp)
    test_samples = []
    for i in range(0, len(all_user_chunks)):
        user = all_user_chunks[i]
        train_size = round(len(user) * TRAIN_PERCENT)
        for chunk in user[train_size:]:
            test_samples.append(build_features_for_chunk(user_id, chunk))
    return test_samples


def main(user_id):
    train_samples = build_train_samples(user_id)
    with open('user1_train_features', 'wb') as fp:
        pickle.dump(train_samples, fp)
    test_samples = build_test_samples(user_id)
    with open('user1_test_features', 'wb') as fp:
        pickle.dump(test_samples, fp)
    #build random forest

if __name__ == "__main__":
    main(1)
