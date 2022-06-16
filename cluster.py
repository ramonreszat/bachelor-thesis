#!/usr/bin/env python3
"""Builds an unsupervised clustering model on an affinity graph.

Usage:
    ./cluster.py --GSVD --graph ./graphs/fused_similarity_network_K=20_gamma=0.5_t=2.gexf
    --sites LUNG --subtypes adenocarcinoma small_cell_carcinoma --emin=0.0

Author:
    Ramón Reszat - 11.03.2022
"""
import pickle
import argparse

import numpy as np
import pandas as pd

import networkx as nx
from tabulate import tabulate
from scipy.sparse import csr_matrix

from sknetwork.clustering import KMeans
from sknetwork.embedding import Spectral, GSVD, PCA

from sknetwork.visualization import svg_graph
from sklearn.metrics.cluster import contingency_matrix as contingency
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.metrics.cluster import adjusted_rand_score as ari


def main(args):
    print('Loading network from',args.graph)
    G = load_network(args.graph, args.sites, min_edge_weight=args.emin)

    y = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
    y = y[y.ccle_hist_subtype_1.isin(args.subtypes)]

    y_labels = y.ccle_hist_subtype_1.values
    y_cat = np.unique(y_labels, return_inverse=True)

    W = nx.to_pandas_adjacency(G)
    W = W[y.index.values].T[y.index.values]
    adjacency = csr_matrix(W, dtype=np.float64)

    if args.GSVD:
        embedding = GSVD(2)
    if args.Spectral:
        embedding = Spectral(n_components=2, decomposition='laplacian')
    if args.PCA:
        embedding = PCA(n_components=2)

    print("KMeans clustering with k={}".format(2))
    kmeans = KMeans(n_clusters=2, embedding_method=embedding)
    y_pred = kmeans.fit_transform(adjacency)
    scores = kmeans.membership_.toarray()

    eval(y_pred, y_labels, y_cat[0])


def load_network(graph_path, primary_site, min_edge_weight=0.0012):
    G = nx.read_gexf(graph_path)

    # select samples in the graph by organ of origin
    selected_nodes = [n for n,v in G.nodes(data=True) if v['primary_site'] in primary_site]
    H = G.subgraph(selected_nodes)

    # create a sparse representation of the network by deleting edges
    selected_edges = [(u,v) for u,v,e in H.edges(data=True) if e['weight'] >= min_edge_weight]
    G = H.edge_subgraph(selected_edges)

    return G


def eval(y_pred, labels, classes):
    print("Contingency matrix:")
    contingency_matrix = pd.DataFrame(contingency(y_pred, labels), index=classes)
    print(tabulate(contingency_matrix, headers='keys', tablefmt='psql'))

    contingency_matrix = contingency(y_pred, labels)
    purity = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
    print("Purity against the labeling: {:.4f}".format(purity))
    print("Adjusted rand index: {:.4f}".format(ari(y_pred, labels)))
    print("Noramlized mutual information: {:.4f}".format(nmi(y_pred, labels)))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--GSVD', action=argparse.BooleanOptionalAction)
    parser.add_argument('--Spectral', action=argparse.BooleanOptionalAction)
    parser.add_argument('--PCA', action=argparse.BooleanOptionalAction)
    parser.add_argument("--graph", help="Graph file in GEXF format with the input data")
    parser.add_argument("--emin", type=float, help="minimum edge weight to consider in the network")
    parser.add_argument("--sites", nargs="+", default=["LUNG"], help="organs of origin of cell line tissues")
    parser.add_argument("--subtypes", nargs="+", default=["adenocarcinoma", "small_cell_carcinoma"], help="cancer histology subtypes")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    main(args)