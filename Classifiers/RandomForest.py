import pickle
from abc import ABC

from sklearn.ensemble import RandomForestClassifier
import numpy as np
from Classifiers.classifier import Classifier
from FeatureExtraction.name_time_features import *
from sklearn.metrics import plot_confusion_matrix, confusion_matrix
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
        error_rate = []
        confusion_matrix_list = []
        # Calculating error for K values between 1 and 40
        fisher_score_threshold = [2000, 3000, 5000, 10000, -1]
        for i in fisher_score_threshold:
            self.write_train_test_sets(user_id, i)
            train_set = pd.read_csv(f'../FileCenter/FeaturesPerUser/user{user_id}_train_features.csv')
            test_set = pd.read_csv(f'../FileCenter/FeaturesPerUser/user{user_id}_test_features.csv')
            # build random forest
            x_train = train_set.iloc[:, :-1]
            x_test = test_set.iloc[:, :-1]
            y_test = test_set['label']
            clf = RandomForestClassifier(n_estimators=NUM_OF_ESTIMATORS)
            clf.fit(x_train, train_set['label'])
            pred_i = clf.predict(x_test)
            cm = confusion_matrix(test_set['label'], pred_i, normalize='true')
            confusion_matrix_list.append(cm)
            if i == -1:
                fisher_score_threshold[4] = len(x_train.columns)
            print(train_set.shape)
            print(pd.Series(pred_i[:431]).value_counts())
            print(pd.Series(pred_i[431:]).value_counts())
            print(np.mean(pred_i != y_test))
            print(f'tp{cm[1][1]}')
            print(f'fp{cm[0][1]}')
            print(f'tn{cm[0][0]}')
            print(f'fn{cm[1][0]}')
            error_rate.append(np.mean(pred_i != y_test))
        self._plot_error_rate(fisher_score_threshold, error_rate)
        self._plot_confusion_matrix(fisher_score_threshold, confusion_matrix_list)

    def _plot_error_rate(self, fisher_score_threshold, error_rate):
        plt.figure()
        plt.plot(fisher_score_threshold, error_rate, color='red', linestyle='dashed', marker='o',
                 markerfacecolor='blue', markersize=10)
        plt.title('error rate for number of features')
        plt.xlabel('k features selected by fisher score')
        plt.ylabel('error rate')
        plt.show()

    def _plot_confusion_matrix(self, fisher_score_threshold, confusion_matrix_list):
        # TPR
        plt.figure()
        plt.plot(fisher_score_threshold, [cm[1][1] for cm in confusion_matrix_list], color='red', linestyle='dashed',
                 marker='o',
                 markerfacecolor='blue', markersize=10)
        plt.title('TPR ')
        plt.xlabel('k features selected by fisher score')
        plt.ylabel('true positive rate')
        plt.show()
        # FPR
        plt.figure()
        plt.plot(fisher_score_threshold, [cm[0][1] for cm in confusion_matrix_list], color='red',
                 linestyle='dashed', marker='o',
                 markerfacecolor='blue', markersize=10)
        plt.title('FPR ')
        plt.xlabel('k features selected by fisher score')
        plt.ylabel('false positive rate')
        plt.show()


if __name__ == '__main__':
    rf = RandomForest()
    rf.write_train_test_sets(0, 10000)
    rf.predict(0)
    #rf.plot_graphs(0)
