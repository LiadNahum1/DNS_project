# -*- coding: utf-8 -*-
"""Attack Detection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-klseWh1hOVjkXmZMP73FcSUnBCEzXKU
"""

import pandas as pd
import pickle
import pickle as pkl

from sklearn.feature_extraction.text import TfidfVectorizer

FILE_NAME = '/content/drive/My Drive/DNS_project/DNS_origional/dnsSummary_user'
FILE_NAME_EXTRACTED = '../DNS_time_name_extracted/dnsSummary_user'
FILE_EXTENSION = '.pcap.csv'
NUMBER_OF_USERS = 15
DAYS = 7
HOURS = 24
EPOCH_DAY = 5


def extract_time(user_data):
    num_days = round(user_data['frame.time_epoch'] / 3600 / 24)
    day = (num_days + EPOCH_DAY) % DAYS + 1
    num_hours = round(user_data['frame.time_epoch'] / 3600)
    hour = num_hours % HOURS + 1
    new_user_data = pd.concat([day, hour, user_data['frame.time_relative'], user_data['dns.qry.name']], axis=1)
    new_user_data.columns = ['day', 'hour', 'relative_time', 'dns_name']
    return new_user_data


def extract_features():
    for i in range(1, NUMBER_OF_USERS + 1):
        user_data = pd.read_csv(FILE_NAME + str(i) + FILE_EXTENSION)
        user_data = user_data.iloc[::2]
        user_data = user_data[['frame.time_epoch', 'frame.time_relative', 'dns.qry.name']]
        new_user_data = extract_time(user_data)
        new_user_data.to_csv(FILE_NAME_EXTRACTED + str(i) + FILE_EXTENSION)


def build_chunk_30_minutes(user_data):
    max_time = user_data['relative_time'].max()
    user_data_list = []
    i = 0
    while i < max_time:
        chunk = user_data.loc[(user_data['relative_time'] > i) & (user_data['relative_time'] < i + 1800)]
        chunk = chunk[['day', 'hour', 'dns_name']].values.tolist()
        if len(chunk) > 0:
            user_data_list.append(chunk)
        i += 1800
    return user_data_list


def build_users_chunks():
    all_users_chunks = []
    for i in range(1, NUMBER_OF_USERS + 1):
        user_data = pd.read_csv(FILE_NAME_EXTRACTED + str(i) + FILE_EXTENSION)
        all_users_chunks.append(build_chunk_30_minutes(user_data))
    with open('all_user_chunks', 'wb') as fp:
        pickle.dump(all_users_chunks, fp)


def build_empty_dictionaries():
    file_all_user_chunks = open('all_user_chunks', 'rb')
    all_user_chunks = pkl.load(file_all_user_chunks)
    file_all_user_chunks.close()
    all_users_dictionaries = []
    for user in all_user_chunks:
        tuples_for_user = {}
        for chunk in user:
            for tuple in chunk:
                if (tuple[1], tuple[2]) not in tuples_for_user:
                    tuples_for_user[(tuple[1], tuple[2])] = 0
        all_users_dictionaries.append(tuples_for_user)
    with open('all_users_hour_name_tuples', 'wb') as fp:
        pickle.dump(all_users_dictionaries, fp)


def tokenizer(s):
    return s.split(' ')


def tfidf_n_grams():
    with open('all_user_chunks', 'rb') as fp:
        all_user_chunks = pkl.load(fp)
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
    tf_idf = pd.DataFrame(denselist, columns=vectorizer.get_feature_names())
    tf_idf.to_csv('tf-idf.csv')

if __name__ == '__main__':
    tfidf_n_grams()

