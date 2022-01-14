import networkx as nx
import numpy as np
import pandas as pd
from pyspark.sql import types
import scipy


def closeness_centrality_scipy(G):
    A = nx.adjacency_matrix(G).tolil()
    D = scipy.sparse.csgraph.floyd_warshall( \
             A, directed=False, unweighted=False)
    n = D.shape[0]
    closeness_centrality = {}
    for r in range(0, n):
        cc = 0.0
        possible_paths = list(enumerate(D[r, :]))
        shortest_paths = dict(filter( \
            lambda x: not x[1] == np.inf, possible_paths))
        total = sum(shortest_paths.values())
        n_shortest_paths = len(shortest_paths) - 1.0
        if total > 0.0 and n > 1:
            s = n_shortest_paths / (n - 1)
            cc = (n_shortest_paths / total) * s
        closeness_centrality[str(r)] = cc
    return closeness_centrality


def node_level_features(
    sparkdf,
    src="src",
    dst="dst",
    block_patch_id="block_patch",
    cluster_id_colname="cluster_id",
):
    ecschema = types.StructType(
        [
            types.StructField("node_id", types.StringType()),
            types.StructField(block_patch_id, types.StringType()),
            types.StructField("degrees", types.DoubleType()),
            types.StructField("clustering_coefficient", types.DoubleType()),
            types.StructField("degree_centrality", types.DoubleType()),
            types.StructField("between_centrality", types.DoubleType()),
            types.StructField("eignen_centrality", types.DoubleType()),
            types.StructField("katz_centrality", types.DoubleType()),
            types.StructField(cluster_id_colname, types.LongType())
        ]
    )
    psrc = src
    pdst = dst
    def udf(pdf: pd.DataFrame) -> pd.DataFrame:
        nxGraph = nx.Graph()
        nxGraph = nx.from_pandas_edgelist(pdf, psrc, pdst)
        degrees = dict(nxGraph.degree())
        clustering_coefficient = nx.clustering(nxGraph)
        degree_centrality = nx.degree_centrality(nxGraph)
        between_centrality = nx.betweenness_centrality(
            nxGraph,
            k=int(min(np.power(nxGraph.number_of_nodes(), 1./2.), 1000.))
        )
        eignen_centrality = nx.eigenvector_centrality(nxGraph, tol=1e-3)
        katz_centrality = nx.katz_centrality(nxGraph, tol=1e-2)
        features = [
            degrees,
            clustering_coefficient,
            degree_centrality,
            between_centrality,
            eignen_centrality,
            katz_centrality
        ]
        features_df = pd.DataFrame.from_records(features).T.reset_index()
        features_df.columns = [
            'node_id',
            'degrees',
            'clustering_coefficient',
            'degree_centrality',
            'between_centrality',
            'eignen_centrality',
            'katz_centrality'
        ]
        cluster_id = pdf[cluster_id_colname][0]
        block_patch = pdf[block_patch_id][0]
        features_df[cluster_id_colname] = cluster_id
        features_df[block_patch_id] = block_patch
        return features_df
    out = sparkdf.groupby([block_patch_id, cluster_id_colname]).applyInPandas(udf, ecschema)
    return out