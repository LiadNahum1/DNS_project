import pickle
from abc import ABC

from sklearn.neighbors import NearestCentroid

from Classifiers.classifier import Classifier
from FeatureExtraction.name_time_features import *


class NearestCentroid(Classifier, ABC):

    def predict(self, user_id):
        train_set = pd.read_csv(f'../FileCenter/FeaturesPerUser/user{user_id}_train_features.csv')
        test_set = pd.read_csv(f'../FileCenter/FeaturesPerUser/user{user_id}_test_features.csv')
        clf = NearestCentroid()
        x_train = train_set.iloc[:, :-1]
        clf.fit(x_train, train_set['label'])
        x_test = test_set.iloc[:, :-1]
        predicted = clf.predict(x_test)
        print(pd.Series(predicted[:431]).value_counts())
        print(pd.Series(predicted[431:]).value_counts())
        with open('../FileCenter/classifiers_predictions/predicted_NearestCentroid', 'wb') as fp:
            pickle.dump(predicted, fp)
