import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle as pkl

fileName = 'all_user_chunks'
USER_COUNT = 15
feature_number_from_tfidf = 5000
# all user chunks = [user[chunk[day,hour,name]]]

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


def get_all_features_of_all_users():
    fileObject2 = open(fileName, 'rb')
    modelInput = pkl.load(fileObject2)
    fileObject2.close()
    tidf_gram = tidf_n_grams(modelInput)
    for i in range(0,USER_COUNT):
        top_user_ngrams = best_ngrams(i, tidf_gram)


if __name__ == "__main__":
    get_all_features_of_all_users()