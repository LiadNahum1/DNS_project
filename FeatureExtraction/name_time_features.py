import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

USER_COUNT = 15


# all user chanks = [user[chank[day,hour,name]]]

def tidf_n_grams(all_user_chanks):
    words_per_user = []
    for i in range(0, USER_COUNT):
        file_of_user = open("FraudedRawData/User" + str(i), "r")
        n_gram_str = ""
        n_gram_1 = file_of_user.readline()[:-1]
        n_gram_2 = file_of_user.readline()[:-1]
        for j in range(2, WORDS_COUNT_PER_SEGMENT * TRAIN_SEGMENT_COUNT):
            n_gram_3 = file_of_user.readline()[:-1]
            n_gram_str = n_gram_str + n_gram_1 + n_gram_2 + n_gram_3 + " "
            n_gram_1 = n_gram_2
            n_gram_2 = n_gram_3
        words_per_user.append(n_gram_str[:-1])
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(words_per_user)
    dense = X.todense()
    denselist = dense.tolist()
    return pd.DataFrame(denselist, columns=vectorizer.get_feature_names())
