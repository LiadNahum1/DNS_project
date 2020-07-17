import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from FeatureExtraction.name_time_features import *


def predict(user_id):
    train_set = pd.read_csv(f'../FileCenter/FeaturesPerUser/user{user_id}_train_features.csv')
    test_set = pd.read_csv(f'../FileCenter/FeaturesPerUser/user{user_id}_test_features.csv')
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(15,), random_state=1000)
    x_train = train_set.iloc[:, :-1]
    clf.fit(x_train, train_set['label'])
    x_test = test_set.iloc[:, :-1]
    predicted = clf.predict(x_test)
    print(pd.Series(predicted[:431]).value_counts())
    print(pd.Series(predicted[431:]).value_counts())
    with open('../FileCenter/predicted_MPL', 'wb') as fp:
        pickle.dump(predicted, fp)


def plot_best_mpl(user_id):
    error = []
    train_set = pd.read_csv(f'../FileCenter/FeaturesPerUser/user{user_id}_train_features.csv')
    test_set = pd.read_csv(f'../FileCenter/FeaturesPerUser/user{user_id}_test_features.csv')
    # build random forest
    x_train = train_set.iloc[:, :-1]
    x_test = test_set.iloc[:, :-1]
    y_test = test_set['label']
    # Calculating error for K values between 1 and 40
    neighbor_tries = [1, 10, 50, 100, 200, 500, 1000, 5000]
    for i in neighbor_tries:
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=(15,), random_state=i)
        clf.fit(x_train, train_set['label'])
        pred_i = clf.predict(x_test)
        error.append(np.mean(pred_i != y_test))
    plt.figure()
    plt.plot(neighbor_tries, error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('error for number of random states')
    plt.xlabel('random states number')
    plt.ylabel('Mean Error')
    plt.show()


if __name__ == "__main__":
    plot_best_mpl(0)
