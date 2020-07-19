from Classifiers.KNearestNeighbors import KNearestNeighbors
from Classifiers.NearestCentroid import NearestCentroid
from Classifiers.NeuralNetwork import NeuralNetwork
from Classifiers.RandomForest import RandomForest, NUM_OF_ESTIMATORS
from FeatureExtraction.analyze_name_time_data import extract_features
from FeatureExtraction.analyze_name_time_data import build_chunk_30_minutes

from Classifiers.SVC import SVC
import pyshark
import pandas as pd
import os

KNEAREST_NEIGHBORS_IND = 0
NEAREST_CENTROID_IND = 1
NEURAL_NETWORK_IND = 2
RANDOM_FOREST_IND = 3
SVC_IND = 4

userIps = [("173.27.225.202", 1), ("173.27.225.197", 2), ("173.27.225.182", 3), ("173.27.225.164", 4),
           ("173.27.225.158", 5), ("173.27.225.153", 6), ("173.27.225.151", 7), ("173.27.225.144", 8),
           ("173.27.225.126", 9), ("173.27.225.118", 10), ("173.27.225.116", 11),
           ("173.27.225.115", 12), ("173.27.225.113", 13), ("173.27.225.111", 14), ("173.27.225.102", 15)]

classifiers = [KNearestNeighbors(), NearestCentroid(), NeuralNetwork(), RandomForest(), SVC()]


def check_ip(x):
    for ip in userIps:
        if x == ip[0]:
            return ip[1]
    raise Exception("Ip not part of the organization")  # Alert admin


def apply_classifier(chunk):
    return RandomForest().predict(chunk)


def create_chunk(user_csv, id):
    extract_features(user_csv)
    user_data = pd.read_csv("user_extracted" + id)
    chunk = build_chunk_30_minutes(user_data)
    return chunk


def alert(id):
    raise Exception("Alert admin" + id)


def check_user(id):
    user_csv = pd.read_csv("user" + id + "csv")
    chunk = create_chunk(user_csv, id)
    ident = apply_classifier(chunk)
    if ident != id:
        alert(id)


def add_to_file(filename, row):
    with open(filename, 'a') as fd:
        fd.write(row)


def add_to_user(pkt):
    curr_ip = pkt['ip.src']
    user = check_ip(curr_ip)
    if user == 1:
        add_to_file("user1.csv", pkt)
    if user == 2:
        add_to_file("user2.csv", pkt)
    if user == 3:
        add_to_file("user3.csv", pkt)
    if user == 4:
        add_to_file("user4.csv", pkt)
    if user == 5:
        add_to_file("user5.csv", pkt)
    if user == 6:
        add_to_file("user6.csv", pkt)
    if user == 7:
        add_to_file("user7.csv", pkt)
    if user == 8:
        add_to_file("user8.csv", pkt)
    if user == 9:
        add_to_file("user9.csv", pkt)
    if user == 10:
        add_to_file("user10.csv", pkt)
    if user == 11:
        add_to_file("user11.csv", pkt)
    if user == 12:
        add_to_file("user12.csv", pkt)
    if user == 13:
        add_to_file("user13.csv", pkt)
    if user == 14:
        add_to_file("user14.csv", pkt)
    if user == 15:
        add_to_file("user15.csv", pkt)


def backup_and_clean_csvs():
    # currently just clean the csvs . possible to
    # create secondaty backup inorder to mantain data
    # for future retraining of the model
    i = 1
    while i <= 15:
        os.remove("user" + i + ".csv")


def check_users():
    check_user(1)
    check_user(2)
    check_user(3)
    check_user(4)
    check_user(5)
    check_user(6)
    check_user(7)
    check_user(8)
    check_user(9)
    check_user(10)
    check_user(11)
    check_user(12)
    check_user(13)
    check_user(14)
    check_user(15)
    backup_and_clean_csvs()


def save_and_check(pkt):
    if pkt.dns.qry_name:
        add_to_user(pkt)
    elif pkt.dns.resp_name:
        add_to_user(pkt)
    check_users()


def recorder():
    # Using pyshark to receive wireshark information
    while True:
        capture = pyshark.LiveCapture(interface='eth0')
        capture.sniff(timeout=1800)  # timeout in seconds ( 30 min = 1800 sec )
        capture.apply_on_packets(save_and_check, timeout=1800)


if __name__ == '__main__':
    recorder()
