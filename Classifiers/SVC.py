import pickle
from abc import ABC

import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC

from Classifiers.classifier import Classifier
from FeatureExtraction.name_time_features import *


class SVCClassifier(Classifier, ABC):

    def predict(self, user_id):
        train_set = pd.read_csv(f'../FileCenter/FeaturesPerUser/user{user_id}_train_features.csv')
        test_set = pd.read_csv(f'../FileCenter/FeaturesPerUser/user{user_id}_test_features.csv')
        clf = SVC(random_state=0)
        x_train = train_set.iloc[:, :-1]
        clf.fit(x_train, train_set['label'])
        x_test = test_set.iloc[:, :-1]
        plot_confusion_matrix(clf, x_test, test_set['label'], normalize='true')  # doctest: +SKIP
        plt.show()
        predicted = clf.predict(x_test)
        with open('../FileCenter/classifiers_predictions/predicted_SVC', 'wb') as fp:
            pickle.dump(predicted, fp)
        return predicted

if __name__ == '__main__':
    nc = SVCClassifier()
    nc.predict(0)