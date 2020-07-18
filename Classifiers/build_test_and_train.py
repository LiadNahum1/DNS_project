from FeatureSelection.fihser_score_selection import fisher_score_selection
from FeatureExtraction.name_time_features import *

TRAIN_PERCENT_OF_OTHER_USERS = 0.1
BELONG_LABEL = 1


def get_train_samples_of_current_user(all_user_chunks, user_id, name_time_features):
    train_samp = []
    user = all_user_chunks[user_id]
    train_size = round(len(user) * TRAIN_PERCENT)
    # get the features for the user that we check belonging to and appand to them belong Label
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
            for chunk in user[-train_size:]:
                features_for_chunk = name_time_features.build_features_for_chunk(user_id, chunk)
                features_for_chunk.append(1 - BELONG_LABEL)
                train_samp.append(features_for_chunk)
    return train_samp


def build_train_samples(user_id, name_time_features):
    with open('../FileCenter/minimzed_data/all_user_chunks', 'rb') as fp:
        all_user_chunks = pkl.load(fp)
    train_samples = get_train_samples_of_current_user(all_user_chunks, user_id, name_time_features)
    train_samples.extend(get_train_samples_of_rest_users(all_user_chunks, user_id, name_time_features))
    return train_samples


# user_id is the user that we check if the chunk belongs to him
def build_test_samples(user_id, name_time_features):
    with open('../FileCenter/minimzed_data/all_user_chunks', 'rb') as fp:
        all_user_chunks = pkl.load(fp)
    test_samples = []
    for i in range(0, len(all_user_chunks)):
        if i != user_id:
            user = all_user_chunks[i]
            train_size = round(len(user) * TRAIN_PERCENT)
            test_size = round(len(user) * TRAIN_PERCENT_OF_OTHER_USERS)
            for chunk in user[train_size:train_size + test_size]:  # 10 percent
                features_for_chunk = name_time_features.build_features_for_chunk(user_id, chunk)
                features_for_chunk.append(1 - BELONG_LABEL)
                test_samples.append(features_for_chunk)
        else:
            user = all_user_chunks[i]
            train_size = round(len(user) * TRAIN_PERCENT)
            for chunk in user[train_size:]:  # 30 percent
                features_for_chunk = name_time_features.build_features_for_chunk(user_id, chunk)
                features_for_chunk.append(BELONG_LABEL)
                test_samples.append(features_for_chunk)
    return test_samples


def write_train_test_sets(user_id):
    name_time_features = NameTimeFeatures()
    features_names = name_time_features.get_features_of_user(user_id)

    features_names.append('label')
    train_samples = build_train_samples(user_id, name_time_features)
    train_set = pd.DataFrame(data=train_samples, columns=features_names)

    test_samples = build_test_samples(user_id, name_time_features)
    test_set = pd.DataFrame(data=test_samples, columns=features_names)

    # write into files
    train_set.to_csv(f'../FileCenter/FeaturesPerUser/user{user_id}_train_features.csv')
    test_set.to_csv(f'../FileCenter/FeaturesPerUser/user{user_id}_test_features.csv')


def write_train_test_sets__with_fisher_score(user_id, fisher_score_threshold):
    name_time_features = NameTimeFeatures()
    features_names = name_time_features.get_features_of_user(user_id)

    features_names.append('label')
    train_samples = build_train_samples(user_id, name_time_features)
    train_set = pd.DataFrame(data=train_samples, columns=features_names)

    test_samples = build_test_samples(user_id, name_time_features)
    test_set = pd.DataFrame(data=test_samples, columns=features_names)

    # features selection by fisher score
    indexes_by_fisher_score = fisher_score_selection(train_set, fisher_score_threshold)
    label_column_index = len(train_set.columns) - 1
    indexes_by_fisher_score.append(label_column_index)
    train_set = train_set.iloc[:, indexes_by_fisher_score]
    test_set = test_set.iloc[:, indexes_by_fisher_score]

    # write into files
    train_set.to_csv(f'../FileCenter/FeaturesPerUser/user{user_id}_train_features.csv')
    test_set.to_csv(f'../FileCenter/FeaturesPerUser/user{user_id}_test_features.csv')

if __name__ == "__main__":
    write_train_test_sets(0)
    print("done")
