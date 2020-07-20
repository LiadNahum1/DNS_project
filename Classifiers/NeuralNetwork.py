import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix
from Classifiers.classifier import Classifier
from FeatureExtraction.name_time_features import *


class NeuralNetwork(Classifier):
    def predict(self, user_id):
        train_set = pd.read_csv(f'../FileCenter/FeaturesPerUser/user{user_id}_train_features.csv')
        test_set = pd.read_csv(f'../FileCenter/FeaturesPerUser/user{user_id}_test_features.csv')
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=(200,))
        x_train = train_set.iloc[:, :-1]
        clf.fit(x_train, train_set['label'])
        x_test = test_set.iloc[:, :-1]
        plot_confusion_matrix(clf, x_test, test_set['label'], normalize='true')  # doctest: +SKIP
        plt.show()
        predicted = clf.predict(x_test)
        with open('../FileCenter/predicted_MPL', 'wb') as fp:
            pickle.dump(predicted, fp)
        return predicted

    # plot best mpl
    def plot_graphs(self, user_id):
        error = []
        train_set = pd.read_csv(f'../FileCenter/FeaturesPerUser/user{user_id}_train_features.csv')
        test_set = pd.read_csv(f'../FileCenter/FeaturesPerUser/user{user_id}_test_features.csv')
        # build random forest
        x_train = train_set.iloc[:, :-1]
        x_test = test_set.iloc[:, :-1]
        y_test = test_set['label']
        # Calculating error for K values between 1 and 40
        neighbor_tries = [1, 15, 50, 100, 200]
        for i in neighbor_tries:
            clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                                hidden_layer_sizes=(i,))
            clf.fit(x_train, train_set['label'])
            pred_i = clf.predict(x_test)
            error.append(np.mean(pred_i != y_test))
        plt.figure()
        plt.plot(neighbor_tries, error, color='red', linestyle='dashed', marker='o',
                 markerfacecolor='blue', markersize=10)
        plt.title('error for number of hidden layers')
        plt.xlabel('number of hidden layers')
        plt.ylabel('Mean Error')
        plt.show()


if __name__ == '__main__':
    nc = NeuralNetwork()
    nc.predict(0)
