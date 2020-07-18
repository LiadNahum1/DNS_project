import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle as pkl

USER_COUNT = 15
TRAIN_PERCENT = 0.7
FEATURE_NUMBER_FROM_TFIDF = 5000


class NameTimeFeatures:
    def __init__(self):
        self.tfidf_grams = pd.read_csv('../FileCenter/tfidf.csv')
        with open('../FileCenter/minimzed_data/all_users_hour_name_tuples', 'rb') as fp:
            self.all_user_hour_name_dict = pkl.load(fp)

    # taking top 1000 ngrams for specific user
    def best_ngrams(self, user_id):
        columns = []
        for i in range(0, USER_COUNT):
            columns.append(f'user_{i}')
        n_grams_tidf_T = pd.DataFrame(data=self.tfidf_grams.T)
        n_grams_tidf_T.columns = columns
        best_ngrams = n_grams_tidf_T.nlargest(FEATURE_NUMBER_FROM_TFIDF, f'user_{user_id}').index.to_list()
        return best_ngrams  # list of ngrams words

    def build_word_dict_dns_name_tfidf(self, best_ngrams):
        ngrams_dict = dict()
        for ngram in best_ngrams:
            ngrams_dict[ngram] = 0
        return ngrams_dict

    # returns a dictinary of the words in dict with the occurance of them in specific segment
    def count_word_occurrence(self, dict, segment):
        for word in segment:
            if word in dict:
                dict[word] = dict[word] + 1
        return dict

    def get_dns_name_tfidf(self, user_id, user_chunks):
        # get the top tfidf values that match to the current user id
        top_user_ngrams = self.best_ngrams(user_id)
        # for each chunk we save all the words in a list and than count how much time each of the top user ngrams appear
        # in this chunk
        words_per_user = []
        for chunk in user_chunks:
            words_per_user.append(self.get_dns_name_tfidf__per_chunk(chunk, top_user_ngrams))
        return words_per_user, top_user_ngrams

    def get_dns_name_and_hour(self, user_hour_name_dict, user_chunks):
        user_hour_name_per_chunk = []
        for chunk in user_chunks:
            user_hour_name_per_chunk.append(self.get_dns_name_and_hour__per_chunk(user_hour_name_dict, chunk))
        return user_hour_name_per_chunk

    def get_dns_name_tfidf_ngrams__per_chunk(self, chunk, best_ngrams):
        words_list = []
        if len(chunk) >= 3:
            n_gram_1 = chunk[0][2]
            n_gram_2 = chunk[1][2]
            for tuple in chunk:
                n_gram_3 = tuple[2]
                words_list.append(n_gram_1 + n_gram_2 + n_gram_3)
                n_gram_1 = n_gram_2
                n_gram_2 = n_gram_3
        dict = self.build_word_dict_dns_name_tfidf(best_ngrams)
        chunk_words_count = list(self.count_word_occurrence(dict, words_list).values())
        return chunk_words_count

    def get_dns_name_tfidf__per_chunk(self, chunk, best_ngrams):
        words_list = []
        for tuple in chunk:
            words_list.append(tuple[2])
        dict = self.build_word_dict_dns_name_tfidf(best_ngrams)
        chunk_words_count = list(self.count_word_occurrence(dict, words_list).values())
        return chunk_words_count

    def get_dns_name_features(self, user_id, chunk):
        # get the top tfidf values that match to the current user id
        top_user_ngrams = self.best_ngrams(user_id)
        return self.get_dns_name_tfidf__per_chunk(chunk, top_user_ngrams)

    def get_dns_name_and_hour__per_chunk(self, user_hour_name_dict, chunk):
        chunk_hour_name_dict = user_hour_name_dict.copy()
        for tuple in chunk:
            if (tuple[1], tuple[2]) in chunk_hour_name_dict:
                chunk_hour_name_dict[(tuple[1], tuple[2])] = chunk_hour_name_dict[(tuple[1], tuple[2])] + 1
        return chunk_hour_name_dict.values()

    def get_dns_name_and_hour_features(self, user_id, chunk):
        return self.get_dns_name_and_hour__per_chunk(self.all_user_hour_name_dict[user_id], chunk)

    def get_features_of_user(self, user_id):
        top_user_ngrams = self.best_ngrams(user_id)
        top_user_ngrams.extend(self.all_user_hour_name_dict[user_id].keys())
        return top_user_ngrams

    def build_features_for_chunk(self, user_id, chunk):
        feature_dns_name_tfidf = self.get_dns_name_features(user_id, chunk)
        features_dns_name_and_hour = self.get_dns_name_and_hour_features(user_id, chunk)
        features_of_chunk = feature_dns_name_tfidf
        features_of_chunk.extend(features_dns_name_and_hour)
        return features_of_chunk



def tokenizer(s):
    return s.split(' ')


def get_train_chunks(all_user_chunks):
    train_user_chunks = []
    for user in all_user_chunks:
        train_user_chunks.append(user[0:round(len(user) * TRAIN_PERCENT)])
    return train_user_chunks


# build tfidf ngrams only on train set
def build_tidf_n_grams():
    with open('../FileCenter/minimzed_data/all_user_chunks', 'rb') as fp:
        all_user_chunks = pkl.load(fp)
    train_user_chunks = get_train_chunks(all_user_chunks)
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

# build tfidf with no ngrams only on train set
def build_tidf():
    with open('../FileCenter/minimzed_data/all_user_chunks', 'rb') as fp:
        all_user_chunks = pkl.load(fp)
    train_user_chunks = get_train_chunks(all_user_chunks)
    words_per_user = []
    for users_chunks in train_user_chunks:
        n_gram_str = ""
        for chunk in users_chunks:
            for tuple in chunk:
                n_gram_str = n_gram_str + tuple[2] + " "
        words_per_user.append(n_gram_str[:-1])
    vectorizer = TfidfVectorizer(tokenizer=tokenizer)
    X = vectorizer.fit_transform(words_per_user)
    dense = X.todense()
    denselist = dense.tolist()
    tf_idf = pd.DataFrame(denselist, columns=vectorizer.get_feature_names())
    tf_idf.to_csv('tfidf.csv')

if __name__ == "__main__":
    build_tidf()
