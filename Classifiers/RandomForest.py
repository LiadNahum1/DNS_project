import pickle
from abc import ABC

from sklearn.ensemble import RandomForestClassifier
import numpy as np
from Classifiers.classifier import Classifier
from FeatureExtraction.name_time_features import *
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

NUM_OF_ESTIMATORS = 500


class RandomForest(Classifier, ABC):

    def predict(self, user_id):
        train_set = pd.read_csv(f'../FileCenter/FeaturesPerUser/user{user_id}_train_features.csv')
        test_set = pd.read_csv(f'../FileCenter/FeaturesPerUser/user{user_id}_test_features.csv')
        # build random forest
        clf = RandomForestClassifier(n_estimators=NUM_OF_ESTIMATORS)
        x_train = train_set.iloc[:, :-1]
        clf.fit(x_train, train_set['label'])
        x_test = test_set.iloc[:, :-1]
        predicted = clf.predict(x_test)
        with open('../FileCenter/classifiers_predictions/predicted_random_forest', 'wb') as fp:
            pickle.dump(predicted, fp)
        plot_confusion_matrix(clf, x_test, test_set['label'], normalize='true')
        plt.show()

    # plot best fisher score threshold for random forest
    def plot_graphs(self, user_id):
        error = []
        # Calculating error for K values between 1 and 40
        fisher_score_threshold = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1]
        for i in fisher_score_threshold:
            self.write_train_test_sets__with_fisher_score(user_id, i)
            train_set = pd.read_csv(f'../FileCenter/FeaturesPerUser/user{user_id}_train_features.csv')
            test_set = pd.read_csv(f'../FileCenter/FeaturesPerUser/user{user_id}_test_features.csv')
            # build random forest
            x_train = train_set.iloc[:, :-1]
            x_test = test_set.iloc[:, :-1]
            y_test = test_set['label']
            clf = RandomForestClassifier(n_estimators=NUM_OF_ESTIMATORS)
            clf.fit(x_train, train_set['label'])
            pred_i = clf.predict(x_test)
            error.append(np.mean(pred_i != y_test))
        plt.figure()
        plt.plot(fisher_score_threshold, error, color='red', linestyle='dashed', marker='o',
                 markerfacecolor='blue', markersize=10)
        plt.title('error rate for fisher score threshold')
        plt.xlabel('fisher score threshold')
        plt.ylabel('error rate')
        plt.show()

if __name__ == '__main__':
    rf = RandomForest()
    rf.plot_graphs()