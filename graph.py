#!/usr/bin/env python3
"""Constructs an affinity graph from a multi-omics dataset.

This script applies similarity network fusion according to Wang et al. (2014) with
parameter K and gamma and affinity network fusion as described in Zhang & Ma (2018).

Run ./graph.py -h to get an overview of the parameters.

Usage:
    ./graph.py --snf -K=20 --gamma=0.5 -t=2 --drugs doxorubicin dasatinib vincristine
    ./graph.py --anf -K=20 --alpha=0.1667 --beta=0.1667 -t=2 --drugs doxorubicin dasatinib vincristine

Author:
    Ramón Reszat - 11.03.2022
"""
import argparse
import numpy as np
import pandas as pd
import networkx as nx

import rpy2.robjects as robjects

import rpy2.robjects as ro
import rpy2.robjects.numpy2ri

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from snf import snf, make_affinity


def main(args):
    print("[AFFINITY GRAPHS]")

    print("Load multi-omics dataset (CCLE)...")
    dataset = load_omics_dataset()
    views = [omics.values for omics in dataset]

    print("Load sample information (CTRP)...")
    drugs = pd.read_csv('./data/drugs.csv')
    cells = pd.read_csv('./data/cells.csv', index_col='CCLE_ID')
    node_table, node_labels = attributes(cells, drugs, args.drugs)

    if args.snf:
        print("Similarity Network Fusion (SNF)")
        print("K:{} gamma:{} t:{}...".format(args.K, args.gamma, args.t))
        Wv_SNF, W_SNF = similarity_networks(views, K=args.K, gamma=args.gamma, t=args.t)

        print("Saving affinity graph...")
        G = affinity_graph(W_SNF, node_labels, attrs=node_table)
        nx.write_gexf(G,"./graphs/fused_similarity_network_K={}_gamma={}_t={}.gexf".format(args.K, args.gamma, args.t))

    elif args.anf:
        r = robjects.r
        rpy2.robjects.numpy2ri.activate()
        r['source']('./R/AffinityNetworkFusion.R')

        print("Affinity Network Fusion (ANF)...")
        print("K:{} alpha:{} beta:{} t:{}...".format(args.K, args.alpha, args.beta, args.t))
        Wv_ANF, W_ANF = affinity_networks(views, K=args.K, alpha=args.alpha, beta=args.beta)

        print("Saving affinity graph...")
        G = affinity_graph(W_ANF, node_labels, attrs=node_table)
        nx.write_gexf(G,"./graphs/fused_affinity_network_K={}_alpha={:.4f}_beta={:.4f}_t={}.gexf".format(args.K, args.alpha, args.beta, args.t))
        
    print("[DONE]")


def similarity_networks(views, K=20, gamma=0.5, t=2):
    W_v = make_affinity(views, metric='euclidean', K=K, mu=gamma)
    W = snf(W_v, K=20, t=t)
    return W_v, W


def affinity_networks(views, K=20, alpha=1/6, beta=1/6, t=2):
    distances = [squareform(pdist(X,'euclidean')) for X in views]
    W_v = [affinity_matrix(D, K, alpha, beta) for D in distances]
    W = anf(W_v, K=20)
    return W_v, W


def affinity_matrix(D, K=20, alpha=1/6, beta=1/6):

    nr,nc = D.shape
    Dr = ro.r.matrix(D, nrow=nr, ncol=nc)
    ro.r.assign("D", Dr)

    affinity_matrix_r = robjects.globalenv['affinity_matrix']

    A = affinity_matrix_r(Dr, K, alpha=alpha, beta=beta)
    return A


def anf(W_v, K):
    ANF_r = robjects.globalenv['ANF']
    # two-step default
    W = ANF_r(W_v, K=K)
    return W


def load_omics_dataset():
    genes = pd.read_csv('./data/omics/genes.csv', index_col='CCLE_ID')
    metabolites = pd.read_csv('./data/omics/metabolites.csv', index_col='CCLE_ID')
    mirnas = pd.read_csv('./data/omics/mirnas.csv', index_col='CCLE_ID')

    return [genes, metabolites, mirnas]


def attributes(cells, drugs, cpd_list):
    # set vertex label of each node to its id 
    node_labels = cells.id.reset_index(drop=True)
    node_attrs = cells.reset_index(drop=True).set_index('id')

    sensitivity = drugs.pivot_table(values='area_under_curve', index='cpd_name', columns='ccl_name')

    info = drugs[['ccl_name', 'ccle_primary_site', 'ccle_primary_hist', 'ccle_hist_subtype_1']]
    info = info.drop_duplicates().set_index('ccl_name')

    cell_attrs = node_attrs.join(info)
    cell_attrs = cell_attrs.drop(columns=['ccle_primary_site'])

    # drug sensitivity from cell viability screens
    drug_attrs = sensitivity.transpose()[cpd_list]

    # combined attributes for all samples
    node_table = cell_attrs.join(drug_attrs)

    # handle NaN value for drug sensitvity and subtype
    node_table = node_table.replace(np.nan,"nan", regex=True)

    return node_table, node_labels


def affinity_graph(A, labels, attrs=None):
    G = nx.from_numpy_array(A)
    G = nx.relabel_nodes(G, labels.to_dict())

    for column in attrs:
        nx.set_node_attributes(G, attrs[column].to_dict(), column)
    return G


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snf', action=argparse.BooleanOptionalAction)
    parser.add_argument('--anf', action=argparse.BooleanOptionalAction)
    parser.add_argument("-K", type=int, help="Constructing affinity graphs using k-nearest neighbors")
    parser.add_argument("--alpha", type=float, help="coefficient for local diameters (ANF)")
    parser.add_argument("--beta", type=float, help="coefficient for pair-wise distance (ANF)")
    parser.add_argument("--gamma", type=float, help="scaling factor gamma for the RBF kernel (SNF)")
    parser.add_argument("-t", type=int, help="t steps in the diffusion process (SNF)")
    parser.add_argument("--drugs", nargs="+", default=["doxorubicin", "dasatinib", "vincristine"], help="CTRP anticancer compounds")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
