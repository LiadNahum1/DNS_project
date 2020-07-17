import pickle

from sklearn.ensemble import RandomForestClassifier
from Classifiers import Build_test_and_train
from FeatureExtraction.name_time_features import *


NUM_OF_ESTIMATORS = 500


def predict(user_id):
    train_set = pd.read_csv(f'../FileCenter/FeaturesPerUser/user{user_id}_train_features.csv')
    test_set = pd.read_csv(f'../FileCenter/FeaturesPerUser/user{user_id}_test_features.csv')
    # build random forest
    clf = RandomForestClassifier(n_estimators=NUM_OF_ESTIMATORS)
    x_train = train_set.iloc[:, :-1]
    clf.fit(x_train, train_set['label'])
    x_test = test_set.iloc[:, :-1]
    predicted = clf.predict(x_test)
    with open('../FileCenter/predicted_random_forest', 'wb') as fp:
        pickle.dump(predicted, fp)


if __name__ == "__main__":
    predict(0)
    with open('../FileCenter/predicted_random_forest', 'rb') as fp:
        predicted = pkl.load(fp)
    print(pd.Series(predicted[:431]).value_counts())
    print(pd.Series(predicted[431:]).value_counts())
