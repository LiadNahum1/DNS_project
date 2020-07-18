import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from Classifiers import build_test_and_train
from FeatureExtraction.name_time_features import *
import pickle


def predict(user_id):
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


def plot_best_number_of_neighbors(user_id):
    error = []
    train_set = pd.read_csv(f'../FileCenter/FeaturesPerUser/user{user_id}_train_features.csv')
    test_set = pd.read_csv(f'../FileCenter/FeaturesPerUser/user{user_id}_test_features.csv')
    # build random forest
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


if __name__ == "__main__":
    predict(0)
