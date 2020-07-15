import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle as pkl

USER_COUNT = 15
TRAIN_PERCENT = 0.7
feature_number_from_tfidf = 5000


def tokenizer(s):
    return s.split(' ')


# build tfidf ngrams only on train set
def build_tidf_n_grams():
    with open('all_user_chunks', 'rb') as fp:
        all_user_chunks = pkl.load(fp)
    train_user_chunks = []
    for user in all_user_chunks:
        train_user_chunks.append(user[0:round(len(user) * TRAIN_PERCENT)])
    words_per_user = []
    for users_chunks in train_user_chunks:
        n_gram_str = ""
        for chunk in users_chunks:
            if len(chunk) >= 3:
                n_gram_1 = chunk[0][2]
                n_gram_2 = chunk[1][2]
                for tuple in chunk:
                    n_gram_3 = tuple[2]
                    n_gram_str = n_gram_str + n_gram_1 + n_gram_2 + n_gram_3 + " "
                    n_gram_1 = n_gram_2
                    n_gram_2 = n_gram_3
        words_per_user.append(n_gram_str[:-1])
    vectorizer = TfidfVectorizer(tokenizer=tokenizer)
    X = vectorizer.fit_transform(words_per_user)
    dense = X.todense()
    denselist = dense.tolist()
    tf_idf = pd.DataFrame(denselist, columns=vectorizer.get_feature_names())
    tf_idf.to_csv('tfidf_ngrams.csv')


# taking top 1000 ngrams for specific user
def best_ngrams(user_id, n_grams_tidf):
    columns = []
    for i in range(0, USER_COUNT):
        columns.append(f'user_{i}')
    n_grams_tidf_T = pd.DataFrame(data=n_grams_tidf.T)
    n_grams_tidf_T.columns = columns
    best_ngrams = n_grams_tidf_T.nlargest(feature_number_from_tfidf, f'user_{user_id}').index.to_list()
    return best_ngrams  # list of ngrams words


def build_word_dict_dns_name_tfidf(best_ngrams):
    ngrams_dict = dict()
    for ngram in best_ngrams:
        ngrams_dict[ngram] = 0
    return ngrams_dict


# returns a dictinary of the words in dict with the occurance of them in specific segment
def count_word_occurrence(dict, segment):
    for word in segment:
        if word in dict:
            dict[word] = dict[word] + 1
    return dict


def get_dns_name_tfidf(user_id, user_chunks):
    tfidf_grams = pd.read_csv('tfidf_ngrams.csv')
    # get the top tfidf values that match to the current user id
    top_user_ngrams = best_ngrams(user_id, tfidf_grams)
    # for each chunk we save all the words in a list and than count how much time each of the top user ngrams appear
    # in this chunk
    words_per_user = []
    for chunk in user_chunks:
        words_per_user.append(get_dns_name_tfidf_ngrams__per_chunk(chunk, top_user_ngrams))
    return words_per_user, top_user_ngrams


def get_dns_name_and_hour(user_hour_name_dict, user_chunks):
    user_hour_name_per_chunk = []
    for chunk in user_chunks:
        user_hour_name_per_chunk.append(get_dns_name_and_hour__per_chunk(user_hour_name_dict, chunk))
    return user_hour_name_per_chunk


def get_dns_name_tfidf_ngrams__per_chunk(chunk, best_ngrams):
    words_list = []
    if len(chunk) >= 3:
        n_gram_1 = chunk[0][2]
        n_gram_2 = chunk[1][2]
        for tuple in chunk:
            n_gram_3 = tuple[2]
            words_list.append(n_gram_1 + n_gram_2 + n_gram_3)
            n_gram_1 = n_gram_2
            n_gram_2 = n_gram_3
    dict = build_word_dict_dns_name_tfidf(best_ngrams)
    chunk_words_count = list(count_word_occurrence(dict, words_list).values())
    return chunk_words_count


def get_dns_name_features(user_id, chunk):
    # get the top tfidf values that match to the current user id
    tfidf_grams = pd.read_csv('tfidf_ngrams.csv')
    top_user_ngrams = best_ngrams(user_id, tfidf_grams)
    return get_dns_name_tfidf_ngrams__per_chunk(chunk, top_user_ngrams)


def get_dns_name_and_hour__per_chunk(user_hour_name_dict, chunk):
    chunk_hour_name_dict = user_hour_name_dict.copy()
    for tuple in chunk:
        chunk_hour_name_dict[(tuple[1], tuple[2])] = chunk_hour_name_dict[(tuple[1], tuple[2])] + 1
    return chunk_hour_name_dict.values()


def get_dns_name_and_hour_features(user_id, chunk):
    with open('all_users_hour_name_tuples', 'rb') as fp:
        all_user_hour_name_dict = pkl.load(fp)
    return get_dns_name_and_hour__per_chunk(all_user_hour_name_dict[user_id], chunk)


def build_train_samples():
    with open('all_user_chunks', 'rb') as fp:
        all_user_chunks = pkl.load(fp)
    with open('all_users_hour_name_tuples', 'rb') as fp:
        all_user_hour_name_dict = pkl.load(fp)
    for i in range(0, USER_COUNT):
        features_dns_name_tfidf, top_dns_names_keys = get_dns_name_tfidf(i, all_user_chunks[i])
        features_dns_name_and_hour = get_dns_name_and_hour(all_user_hour_name_dict[i], all_user_chunks[i])
        columns = top_dns_names_keys
        columns.extend(all_user_hour_name_dict[i].keys())
        features_per_chunk = features_dns_name_tfidf
        for chunk_index in range(0, len(features_per_chunk)):
            features_per_chunk[chunk_index].extend(features_dns_name_and_hour[chunk_index])
        features_for_user_i = pd.DataFrame(data=features_per_chunk, columns=columns)
        print(features_for_user_i)
        features_for_user_i.to_csv(f'FeaturesPerUser/features_user_{i + 1}.csv')


def build_features_for_chunk(user_id, chunk):
    feature_dns_name_tfidf = get_dns_name_features(user_id, chunk)
    features_dns_name_and_hour = get_dns_name_and_hour(user_id, chunk)
    features_of_chunk = feature_dns_name_tfidf
    for chunk_index in range(0, len(features_of_chunk)):
        features_of_chunk[chunk_index].extend(features_dns_name_and_hour[chunk_index])
    return features_of_chunk


if __name__ == "__main__":
    build_train_samples()
