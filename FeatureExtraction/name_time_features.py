import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle as pkl

USER_COUNT = 15
feature_number_from_tfidf = 5000


# returns a dictinary of the words in dict with the occurance of them in specific segment
def count_word_occurrence(dict, segment):
    for word in segment:
        if word in dict:
            dict[word] = dict[word] + 1
    return dict


def tokenizer(s):
    return s.split(' ')


def tidf_n_grams(all_user_chunks):
    words_per_user = []
    for users_chunks in all_user_chunks:
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
    return pd.DataFrame(denselist, columns=vectorizer.get_feature_names())


# taking top 1000 ngrams for specific user
def best_ngrams(user_id, n_grams_tidf):
    columns = []
    for i in range(0, USER_COUNT):
        columns.append(f'user_{i}')
    n_grams_tidf_T = pd.DataFrame(data=n_grams_tidf.T)
    n_grams_tidf_T.columns = columns
    best_ngrams = n_grams_tidf_T.nlargest(feature_number_from_tfidf, f'user_{user_id}').index.to_list()
    return best_ngrams #list of ngrams words


def build_word_dict_dns_name_tfidf(best_ngrams):
    ngrams_dict = dict()
    for ngram in best_ngrams:
        ngrams_dict[ngram] = 0
    return ngrams_dict


def get_dns_name_tfidf(user_id, tfidf_grams, user_chunks):
    # get the top tfidf values that match to the current user id
    top_user_ngrams = best_ngrams(user_id, tfidf_grams)
    # for each chunk we save all the words in a list and than count hom much time each of the top user ngrams appear
    # in this chunk
    words_per_user = []
    for chunk in user_chunks:
        words_list = []
        if len(chunk) >= 3:
            n_gram_1 = chunk[0][2]
            n_gram_2 = chunk[1][2]
            for tuple in chunk:
                n_gram_3 = tuple[2]
                words_list.append(n_gram_1 + n_gram_2 + n_gram_3)
                n_gram_1 = n_gram_2
                n_gram_2 = n_gram_3
        dict = build_word_dict_dns_name_tfidf(top_user_ngrams)
        chunk_words_count = list(count_word_occurrence(dict, words_list).values())
        words_per_user.append(chunk_words_count)
    return words_per_user, top_user_ngrams


def get_dns_name_and_hour(user_hour_name_dict, user_chunks):
    user_hour_name_per_chunk = []
    for chunk in user_chunks:
        chunk_hour_name_dict = user_hour_name_dict.copy()
        for tuple in chunk:
            chunk_hour_name_dict[(tuple[1], tuple[2])] = chunk_hour_name_dict[(tuple[1], tuple[2])] + 1
        user_hour_name_per_chunk.append(list(chunk_hour_name_dict.values()))
    return user_hour_name_per_chunk


def get_all_features_of_all_users():
    with open('all_user_chunks', 'rb') as fp:
        all_user_chunks = pkl.load(fp)
    with open('all_users_hour_name_tuples', 'rb') as fp:
        all_user_hour_name_dict = pkl.load(fp)
    # build the tfidf of all the users with 3 ngrams
    tidf_gram = pd.read_csv('tf-idf.csv')
    for i in range(0, USER_COUNT):
        features_dns_name_tfidf, dns_names_keys = get_dns_name_tfidf(i, tidf_gram, all_user_chunks[i])
        features_dns_name_and_hour = get_dns_name_and_hour(all_user_hour_name_dict[i], all_user_chunks[i])
        columns = dns_names_keys
        columns.extend(all_user_hour_name_dict[i].keys())
        features_per_chunk = features_dns_name_tfidf
        for chunk_index in range(0, len(features_per_chunk)):
            features_per_chunk[chunk_index].extend(features_dns_name_and_hour[chunk_index])
        features_for_user_i = pd.DataFrame(data=features_per_chunk, columns=columns)
        print(features_for_user_i)
        features_for_user_i.to_csv(f'features_user_{i+1}.csv')

if __name__ == "__main__":
    get_all_features_of_all_users()
