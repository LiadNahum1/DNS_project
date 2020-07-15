import pickle
import pickle as pkl

from FeatureExtraction.name_time_features import build_features_for_chunk, get_features_of_user, TRAIN_PERCENT

TRAIN_PERCENT_OF_OTHER_USERS = 0.1
BELONG_LABEL = 1


def get_train_samples_of_current_user(all_user_chunks, user_id):
    train_samp = []
    user = all_user_chunks[user_id]
    train_size = round(len(user) * TRAIN_PERCENT)
    for chunk in user[:train_size]:
        features_for_chunk = build_features_for_chunk(user_id, chunk)
        train_samp.append(features_for_chunk.append(BELONG_LABEL))
    return train_samp


def get_train_samples_of_rest_users(all_user_chunks, user_id):
    train_samp = []
    for i in range(0, len(all_user_chunks)):
        if i != user_id:
            user = all_user_chunks[i]
            train_size = round(len(user) * TRAIN_PERCENT_OF_OTHER_USERS)
            for chunk in user[:train_size]:
                features_for_chunk = build_features_for_chunk(user_id, chunk)
                train_samp.append(features_for_chunk.append(1 - BELONG_LABEL))
    return train_samp


def build_train_samples(user_id):
    with open('all_user_chunks', 'rb') as fp:
        all_user_chunks = pkl.load(fp)
    train_samples = []
    train_samples = get_train_samples_of_current_user(all_user_chunks, user_id)
    train_samples.append(get_train_samples_of_rest_users(all_user_chunks, user_id))
    return train_samples


# user_id is the user that we check if the chunk belongs to him
def build_test_samples(user_id):
    with open('all_user_chunks', 'rb') as fp:
        all_user_chunks = pkl.load(fp)
    test_samples = []
    for i in range(0, len(all_user_chunks)):
        user = all_user_chunks[i]
        train_size = round(len(user) * TRAIN_PERCENT)
        for chunk in user[train_size:]:
            features_for_chunk = build_features_for_chunk(user_id, chunk)
            if i != user_id:
                test_samples.append(features_for_chunk.append(1 - BELONG_LABEL))
            else:
                test_samples.append(features_for_chunk.append(BELONG_LABEL))
    return test_samples


def main(user_id):
    features_names = get_features_of_user(user_id)
    train_samples = build_train_samples(user_id)
    train_set = pd.DataFrame(pd=train_samples, columns=[features_names, 'label'])
    print(train_set)
    train_set.to_csv('user1_train_features.csv')
    test_samples = build_test_samples(user_id)
    test_set = pd.DataFrame(pd=test_samples, columns=features_names)
    print(test_set)
    test_set.to_csv('user1_test_features.csv')
    # build random forest


if __name__ == "__main__":
    main(1)
