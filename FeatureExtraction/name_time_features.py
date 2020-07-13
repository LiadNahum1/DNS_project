import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle as pkl

fileName = 'all_user_chunks'
USER_COUNT = 15
feature_number_from_tfidf = 5000


# returns a dictinary of the words in dict with the occurance of them in specific segment
def count_word_occurrence(dict, segment):
    for word in segment:
        if word in dict:
            dict[word] = dict[word] + 1
    return dict


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
    vectorizer = TfidfVectorizer()
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
    return best_ngrams


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
    return words_per_user


def get_dns_name_and_hour(user_id, user_chunks):
    user_segments = []
    file_of_user = open("FraudedRawData/User" + str(user_id), "r")
    for i in range(0, SEGMENT_COUNT):
        lines = []
        for j in range(0, WORDS_COUNT_PER_SEGMENT):
            lines.append(file_of_user.readline()[:-1])
        dict = build_word_dict_feature_1()
        segment_words_count = list(count_word_occurrence(dict, lines).values())
        user_segments.append(segment_words_count)
    return user_segments
    
    word_and_hours_dict = {}
    for chunk in user_chunks:
        for tuple in chunk:
            word_and_hours_dict[(tuple[1], tuple[2])] = word_and_hours_dict[(tuple[1] , tuple[2])] + 1






def get_all_features_of_all_users():
    fileObject2 = open(fileName, 'rb')
    all_user_chunks = pkl.load(fileObject2)
    fileObject2.close()
    all_features_of_all_users = []
    # build the tfidf of all the users with 3 ngrams
    tidf_gram = tidf_n_grams(all_user_chunks)
    for i in range(0, USER_COUNT):
        features_dns_name_tfidf = get_dns_name_tfidf(i, tidf_gram, all_user_chunks[i])
        features_dns_name_and_hour = get_dns_name_and_hour(i, all_user_chunks[i])


if __name__ == "__main__":
    get_all_features_of_all_users()
