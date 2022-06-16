#!/usr/bin/env python3
"""Cleans the CCLE data and constructs a multi-omics dataset.

This script loads to raw omics measurements from the CCLE2019 study
and selects all CCLE cell lines for which all methods are available.
Then quantile normalization is applied on the gene expression
profile in the first view.

Usage:
    ./omics.py ./data/CCLE2019 ./data/omics

Author:
    Ramón Reszat - 08.05.2022
"""
import argparse
import numpy as np
import pandas as pd

def main(args):
    print('[Load CCLE 2019]')
    views, samples = load_ccle()

    print('Cleaning Datasets...')
    views, samples = clean_filter(views, samples)
    print('Quantile Normalization..')
    views[0] = views[0].pipe(quantile_normalize)

    print('Saving X^(v)')
    save(views, samples)
    print('[Saved]')

def load_ccle():
    # load gene expression in reads per kilobase million (rpkm) values
    genes = pd.read_csv('./data/CCLE2019/CCLE_RNAseq_genes_rpkm_20180929.gct', sep='\t')
    # load metabolite levels from liquid chromatography-mass spectrometry (LC-MS)
    metabolites = pd.read_csv('./data/CCLE2019/CCLE_metabolomics_20190502.csv')
    # load microRNA expression levels from qPCR measurements
    micrornas = pd.read_csv('./data/CCLE2019/CCLE_miRNA_20181103.gct', sep='\t')

    cell_lines = pd.DataFrame(
    np.char.split(genes.columns[2:].tolist(), sep='_', maxsplit=1).tolist(),
    columns=["id","primary_site"])

    views = [genes, metabolites, micrornas]
    return views, cell_lines

def clean_filter(views, samples):
    genes, metabolites, micrornas = views

    # filter out cell lines that are not in all of the data frames below
    samples = samples[(samples.id + '_' + samples.primary_site).isin(metabolites.CCLE_ID)]
    samples = samples[(samples.id + '_' + samples.primary_site).isin(micrornas.columns)]

    # The CCLE_ID is split into sample id and primary site
    samples['CCLE_ID'] = samples.id + "_" + samples.primary_site

    genes = genes[np.append((samples.id + '_' + samples.primary_site).values, 'Description')]
    genes = genes.set_index('Description')
    genes.index.names = ['Genes']
    genes = genes.transpose().sort_index()
    genes.index = genes.index.rename('CCLE_ID')

    metabolites = metabolites.loc[metabolites['CCLE_ID'].isin((samples.id + '_' + samples.primary_site).values)]
    metabolites = metabolites.sort_values('CCLE_ID')
    metabolites = metabolites.set_index('CCLE_ID')
    del metabolites['DepMap_ID']

    micrornas = micrornas.drop(columns=['Name']).set_index('Description')
    micrornas.index.names = ['miRNA']
    micrornas = micrornas.transpose()
    micrornas.index.names = ['CCLE_ID']
    micrornas = micrornas.loc[(samples.id + '_' + samples.primary_site).values].sort_index()

    return [genes, metabolites, micrornas], samples

def quantile_normalize(genes):
    # gene view index
    df = genes.transpose()

    # aggregate duplicates
    df = df.groupby(df.index).mean()

    # quantile normalization
    rank_mean = df.stack().groupby(df.rank(method='first').stack().astype(int)).mean()
    df = df.rank(method='min').stack().astype(int).map(rank_mean).unstack()

    # sample view index
    genes = df.transpose()

    return genes

def save(views, samples):
    genes, metabolites, micrornas = views
    samples.to_csv('./data/cells.csv', index=False)
    genes.to_csv('./data/omics/genes.csv')
    metabolites.to_csv('./data/omics/metabolites.csv')
    micrornas.to_csv('./data/omics/mirnas.csv')

def parse_arguments():
    parser = argparse.ArgumentParser()

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    main(args)