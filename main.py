"""
Dmytro Bohynskyi 171699

usage: main.py [-h] [-n N] [-p] [-l L] [-save] [-km] [-kr] [-som] [-f] [-u {S1,Breast}]

List the options:

optional arguments:
  -h, --help      show this help message and exit
  -n N            number of clusters, default [2], max 15
  -p              Plot results.
  -l L            Learn neural network, epochs number
  -save           Save dana in csv
  -km, --kmeans   Use K-means algorithm
  -anf, --anfis   Use Anfis algorithm
  -kr, --keras    Use neural network from keras libraries
  -som, --som     Use SOM algorithm - Self-organizing feature map
  -f, --fuzzy     Use SOM algorithm - Self-organizing feature map
  -u {S1,Breast}  Data: S1 - get data with
                  'http://cs.joensuu.fi/sipu/datasets/s1.txt'; Breast - get
                  data with 'http://cs.joensuu.fi/sipu/datasets/breast.txt'
                  ;Default: S1
"""

import argparse
import io

import pandas as pd
import requests
from numpy import genfromtxt

from scrips.K_means import K_means
from scrips.Network import AI

URL = {
    "S1": [r'http://cs.joensuu.fi/sipu/datasets/s1.txt', "    ", "models/K_means_15_2.csv"],
    "Breast": [r'http://cs.joensuu.fi/sipu/datasets/breast.txt', " ", "models/K_means_2_9.csv"]
}


def get_data(url):
    url, space, path = URL.get(url)
    r = requests.get(url)
    # read data
    i = io.StringIO(r.text)
    data = pd.read_csv(i, sep=space, header=None)
    data = data.to_numpy()

    return url, space, path, data


def k_means(data, arg):
    # call the initialize function to get the centroids
    k_object = K_means(data, classes_num=arg.n, plot=arg.p)
    k_object.k_means(save=arg.save)
    # start of segmentation
    k_object.plot() if arg.p else None


def fuzzy(data, arg):
    # call the initialize function to get the centroids
    k_object = K_means(data, classes_num=arg.n, plot=arg.p)
    # start of segmentation
    k_object.fuzzy()
    k_object.plot() if arg.p else None


def keras(path):
    ai_object = AI()
    ai_object.read_db(path)

    ai_object.machine_learning(epochs=arg.l) if arg.l else None
    # start of segmentation
    ai_object.classification(algorithm="keras") if arg.keras else None
    ai_object.plot(algorithm="keras") if arg.p and arg.keras else None


def som(path):
    ai_object = AI()
    ai_object.read_db(path)
    # start of segmentation
    ai_object.classification(algorithm="som") if arg.som else None
    ai_object.plot(algorithm="som") if arg.p and arg.som else None


def anfis(data, arg):
    types = genfromtxt('models/anfis_clusters.txt')

    k_object = K_means(data, classes_num=int(max(types)), plot=arg.p, data_type=types)
    k_object.clustering_empty()
    # start of segmentation
    k_object.plot(label="Anfis") if arg.p else None


def main(arg):
    # get url and data with this url
    url, space, path, data = get_data(arg.u)

    if arg.kmeans:
        k_means(data, arg)
    if arg.fuzzy:
        fuzzy(data, arg)
    if arg.anfis:
        anfis(data, arg)
    if arg.som:
        keras(path)
    if arg.som:
        keras(path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='List the options:')

    # Optionals
    parser.add_argument('-n', help="number of clusters, default [2], max 15", nargs=1, type=int, default=2)
    parser.add_argument('-p', help="Plot results.", action="store_true")
    parser.add_argument('-l', help="Learn neural network, epochs number",
                        nargs=1, type=int, required=False)
    parser.add_argument('-save', help="Save dana in csv", action="store_true")

    parser.add_argument("-km", "--kmeans", help="Use K-means algorithm", action="store_true")
    parser.add_argument("-anf", "--anfis", help="Use Anfis algorithm", action="store_true")
    parser.add_argument("-kr", "--keras", help="Use neural network from keras libraries", action="store_true")
    parser.add_argument("-som", "--som", help="Use SOM algorithm - Self-organizing feature map", action="store_true")
    parser.add_argument("-f", "--fuzzy", help="Use SOM algorithm - Self-organizing feature map", action="store_true")

    parser.add_argument("-u", help="Data:  S1 - get data with 'http://cs.joensuu.fi/sipu/datasets/s1.txt';"
                                   "       Breast - get data with 'http://cs.joensuu.fi/sipu/datasets/breast.txt' ;"
                                   "Default: S1",
                        nargs=1, choices=["S1", 'Breast'], default='S1')

    arg = parser.parse_args()
    if type(arg.u) is list:
        arg.u = arg.u[0]
    if type(arg.l) is list:
        arg.l = arg.l[0]
    if type(arg.n) is list:
        arg.n = arg.n[0]

    main(arg)
