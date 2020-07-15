from sklearn.ensemble import RandomForestClassifier

from FeatureExtraction.name_time_features import *

TRAIN_PERCENT_OF_OTHER_USERS = 0.1
BELONG_LABEL = 1
NUM_OF_ESTIMATORS = 100


def get_train_samples_of_current_user(all_user_chunks, user_id,name_time_features):
    train_samp = []
    user = all_user_chunks[user_id]
    train_size = round(len(user) * TRAIN_PERCENT)
    for chunk in user[:train_size]:
        features_for_chunk = name_time_features.build_features_for_chunk(user_id, chunk)
        train_samp.append(features_for_chunk.append(BELONG_LABEL))
    return train_samp


def get_train_samples_of_rest_users(all_user_chunks, user_id, name_time_features):
    train_samp = []
    for i in range(0, len(all_user_chunks)):
        if i != user_id:
            user = all_user_chunks[i]
            train_size = round(len(user) * TRAIN_PERCENT_OF_OTHER_USERS)
            for chunk in user[:train_size]:
                features_for_chunk = name_time_features.build_features_for_chunk(user_id, chunk)
                train_samp.append(features_for_chunk.append(1 - BELONG_LABEL))
    return train_samp


def build_train_samples(user_id, name_time_features):
    with open('../FileCenter/all_user_chunks', 'rb') as fp:
        all_user_chunks = pkl.load(fp)
    train_samples = get_train_samples_of_current_user(all_user_chunks, user_id, name_time_features)
    train_samples.append(get_train_samples_of_rest_users(all_user_chunks, user_id, name_time_features))
    return train_samples


# user_id is the user that we check if the chunk belongs to him
def build_test_samples(user_id, name_time_features):
    with open('../FileCenter/all_user_chunks', 'rb') as fp:
        all_user_chunks = pkl.load(fp)
    test_samples = []
    for i in range(0, len(all_user_chunks)):
        user = all_user_chunks[i]
        train_size = round(len(user) * TRAIN_PERCENT)
        for chunk in user[train_size:]:
            test_samples.append(name_time_features.build_features_for_chunk(user_id, chunk))
    return test_samples


def write_train_test_sets(user_id):
    name_time_features = NameTimeFeatures()
    features_names = name_time_features.get_features_of_user(user_id)
    train_samples = build_train_samples(user_id, name_time_features)
    train_set = pd.DataFrame(pd=train_samples, columns=[features_names, 'label'])
    print(train_set)
    train_set.to_csv('../FileCenter/FeaturesUser/user1_train_features.csv')
    test_samples = build_test_samples(user_id, name_time_features)
    test_set = pd.DataFrame(pd=test_samples, columns=features_names)
    test_set.to_csv('../FileCenter/FeaturesUser/user1_test_features.csv')


def main(user_id):
    train_set = pd.read_csv(f'../FileCenter/FeaturesUser/user{user_id}_train_features.csv')
    test_set = pd.read_csv(f'../FileCenter/FeaturesUser/user{user_id}_test_features.csv')
    # build random forest
    clf = RandomForestClassifier(n_estimators=NUM_OF_ESTIMATORS)
    train_set = train_set[:, :-1]
    print(train_set)
    clf.fit(train_set, train_set['label'])
    predicted = clf.predict(test_set)
    print(predicted)


if __name__ == "__main__":
    write_train_test_sets(1)
