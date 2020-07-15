import pickle

from sklearn.ensemble import RandomForestClassifier

from FeatureExtraction.name_time_features import *

TRAIN_PERCENT_OF_OTHER_USERS = 0.1
BELONG_LABEL = 1
NUM_OF_ESTIMATORS = 500


def get_train_samples_of_current_user(all_user_chunks, user_id,name_time_features):
    train_samp = []
    user = all_user_chunks[user_id]
    train_size = round(len(user) * TRAIN_PERCENT)
    print(f'user train size{train_size}')
    for chunk in user[:train_size]:
        features_for_chunk = name_time_features.build_features_for_chunk(user_id, chunk)
        features_for_chunk.append(BELONG_LABEL)
        train_samp.append(features_for_chunk)
    return train_samp


def get_train_samples_of_rest_users(all_user_chunks, user_id, name_time_features):
    train_samp = []
    for i in range(0, len(all_user_chunks)):
        if i != user_id:
            user = all_user_chunks[i]
            train_size = round(len(user) * TRAIN_PERCENT_OF_OTHER_USERS)
            print(f'other users train size{train_size}')
            for chunk in user[:train_size]:
                features_for_chunk = name_time_features.build_features_for_chunk(user_id, chunk)
                features_for_chunk.append(1 - BELONG_LABEL)
                train_samp.append(features_for_chunk)
    return train_samp


def build_train_samples(user_id, name_time_features):
    with open('../FileCenter/all_user_chunks', 'rb') as fp:
        all_user_chunks = pkl.load(fp)
    train_samples = get_train_samples_of_current_user(all_user_chunks, user_id, name_time_features)
    train_samples.extend(get_train_samples_of_rest_users(all_user_chunks, user_id, name_time_features))
    print(f'train set size {len(train_samples)}')
    return train_samples


# user_id is the user that we check if the chunk belongs to him
def build_test_samples(user_id, name_time_features):
    with open('../FileCenter/all_user_chunks', 'rb') as fp:
        all_user_chunks = pkl.load(fp)
    test_samples = []
    for i in range(0, len(all_user_chunks)):
        if i != user_id:
            user = all_user_chunks[i]
            train_size = round(len(user) * TRAIN_PERCENT)
            test_size = round(len(user) * TRAIN_PERCENT_OF_OTHER_USERS)
            for chunk in user[train_size:train_size+test_size]: #10 percent
                test_samples.append(name_time_features.build_features_for_chunk(user_id, chunk))
        else:
            user = all_user_chunks[i]
            train_size = round(len(user) * TRAIN_PERCENT)
            print(len(user[train_size:]))
            for chunk in user[train_size:]: #30 percent
                test_samples.append(name_time_features.build_features_for_chunk(user_id, chunk))
    return test_samples


def write_train_test_sets(user_id):
    name_time_features = NameTimeFeatures()
    features_names = name_time_features.get_features_of_user(user_id)
    print(len(features_names))
    test_samples = build_test_samples(user_id, name_time_features)
    test_set = pd.DataFrame(data=test_samples, columns=features_names)
    test_set.to_csv(f'../FileCenter/FeaturesPerUser/user1_test_features.csv')
    print('test done')
    #features_names.append('label')
    #train_samples = build_train_samples(user_id, name_time_features)
    #train_set = pd.DataFrame(data=train_samples, columns=features_names)
    #print(train_set)
    #train_set.to_csv(f'../FileCenter/FeaturesPerUser/user1_train_features.csv')
    #print('train done')



def main(user_id):
    train_set = pd.read_csv(f'../FileCenter/FeaturesPerUser/user{user_id}_train_features.csv')
    print('done')
    test_set = pd.read_csv(f'../FileCenter/FeaturesPerUser/user{user_id}_test_features.csv')
    # build random forest
    clf = RandomForestClassifier(n_estimators=NUM_OF_ESTIMATORS)
    x_train = train_set.iloc[:, :-1]
    print(x_train)
    clf.fit(x_train, train_set['label'])
    predicted = clf.predict(test_set)
    with open('../FileCenter/predicted', 'wb') as fp:
        pickle.dump(predicted, fp)
    predicted1 = predicted[:431]
    predicted2 = print(predicted[431:])
    print(len(predicted))


if __name__ == "__main__":
    write_train_test_sets(1)
    #main(1)
    #with open('../FileCenter/predicted', 'rb') as fp:
     #   predicted = pkl.load(fp)
    #print(predicted)
