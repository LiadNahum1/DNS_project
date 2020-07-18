import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from Classifiers.classifier import Classifier
from FeatureExtraction.name_time_features import *
import pickle


class KNearestNeighbors(Classifier):

    def predict(self, user_id):
        train_set = pd.read_csv(f'../FileCenter/FeaturesPerUser/user{user_id}_train_features.csv')
        test_set = pd.read_csv(f'../FileCenter/FeaturesPerUser/user{user_id}_test_features.csv')
        clf = KNeighborsClassifier(n_neighbors=100)
        x_train = train_set.iloc[:, :-1]
        clf.fit(x_train, train_set['label'])
        x_test = test_set.iloc[:, :-1]
        predicted = clf.predict(x_test)
        print(pd.Series(predicted[:431]).value_counts())
        print(pd.Series(predicted[431:]).value_counts())
        with open('../FileCenter/classifiers_predictions/predicted_KNeighbors', 'wb') as fp:
            pickle.dump(predicted, fp)

    #plot best number of neighbors
    def plot_graphs(self, user_id):
        error = []
        train_set = pd.read_csv(f'../FileCenter/FeaturesPerUser/user{user_id}_train_features.csv')
        test_set = pd.read_csv(f'../FileCenter/FeaturesPerUser/user{user_id}_test_features.csv')
        x_train = train_set.iloc[:, :-1]
        x_test = test_set.iloc[:, :-1]
        y_test = test_set['label']
        # Calculating error for K values between 1 and 40
        neighbor_tries = [1, 10, 50, 100, 500]
        for i in neighbor_tries:
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(x_train, train_set['label'])
            pred_i = knn.predict(x_test)
            error.append(np.mean(pred_i != y_test))
        plt.figure()
        plt.plot(neighbor_tries, error, color='red', linestyle='dashed', marker='o',
                 markerfacecolor='blue', markersize=10)
        plt.title('Error Rate K Value')
        plt.xlabel('K Value')
        plt.ylabel('Mean Error')
        plt.show()