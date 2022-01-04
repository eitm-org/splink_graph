import pyspark
import networkx as nx
import pandas as pd
from pyspark.sql.types import (
    LongType,
    StringType,
    FloatType,
    IntegerType,
    DoubleType,
    StructType,
    StructField,
)
import pyspark.sql.functions as f
from pyspark.sql.functions import pandas_udf, PandasUDFType

from networkx.algorithms.centrality import eigenvector_centrality


def node_level_features(
    sparkdf,
    src="src",
    dst="dst",
    cluster_id_colname="cluster_id",
    # patch_id_colname="PATCH",
    # block_id_colname="BLOCK",
):

    """
    Args:
        sparkdf: imput edgelist Spark DataFrame
        src: src column name
        dst: dst column name
        distance_colname: distance column name
        cluster_id_colname: Graphframes-created connected components created cluster_id
    Returns:
        node_id:
        node_degree
        cluster_id: cluster_id corresponding to the node_id

example input spark dataframe
|src|dst|weight|cluster_id|distance|
|---|---|------|----------|--------|
|  f|  d|  0.67|         0|   0.329|
|  f|  g|  0.34|         0|   0.659|
|  b|  c|  0.56|8589934592|   0.439|
|  g|  h|  0.99|         0|   0.010|
|  a|  b|   0.4|8589934592|     0.6|
|  h|  i|   0.5|         0|     0.5|
|  h|  j|   0.8|         0|   0.199|
|  d|  e|  0.84|         0|   0.160|
|  e|  f|  0.65|         0|    0.35|
example output spark dataframe
|node_id|   degree          |cluster_id|
|-------|-------------------|----------|
|   b   |  0.707106690085642|8589934592|
|   c   | 0.5000000644180599|8589934592|
|   a   | 0.5000000644180599|8589934592|
|   f   | 0.5746147732828122|         0|
|   d   | 0.4584903903420785|         0|
|   g   |0.37778352393858183|         0|
|   h   |0.27663243805676946|         0|
|   i   |0.12277029263709134|         0|
|   j   |0.12277029263709134|         0|
|   e   | 0.4584903903420785|         0|
    """
    ecschema = StructType(
        [
            StructField("node_id", StringType()),
            StructField("eignen_centrality", DoubleType()),
            StructField("katz_centrality", DoubleType()),
            StructField("between_centrality", DoubleType()),
            StructField("degree_centrality", DoubleType()),
            StructField("degrees", DoubleType()),
            StructField(cluster_id_colname, LongType())
        ]
    )
    psrc = src
    pdst = dst
    @pandas_udf(ecschema, PandasUDFType.GROUPED_MAP)
    def udf(pdf: pd.DataFrame) -> pd.DataFrame:
        nxGraph = nx.Graph()
        nxGraph = nx.from_pandas_edgelist(pdf, psrc, pdst)
        degrees = dict(nxGraph.degree())
        eignen_centrality = nx.eigenvector_centrality(nxGraph, tol=1e-3)
        katz_centrality = nx.katz_centrality(nxGraph, tol=1e-2)
        between_centrality = nx.betweenness_centrality(nxGraph)
        degree_centrality = nx.degree_centrality(nxGraph)
        features = [
            eignen_centrality,
            katz_centrality,
            between_centrality,
            degree_centrality,
            degrees,
        ]
        features_df = pd.DataFrame.from_records(features).T.reset_index()
        features_df.columns = [
            'node_id',
            'eignen_centrality',
            'katz_centrality',
            'between_centrality',
            'degree_centrality',
            'degrees'
        ]
        cluster_id = pdf[cluster_id_colname][0]
        # patch_id = pdf[patch_id_colname][0]
        # block_id = pdf[block_id_colname][0]
        features_df[cluster_id_colname] = cluster_id
        # features_df[patch_id_colname] = patch_id
        # features_df[block_id_colname] = block_id
        return features_df
    out = sparkdf.groupby([cluster_id_colname]).apply(udf)
    return out



def eigencentrality(
    sparkdf, src="src", dst="dst", cluster_id_colname="cluster_id",
):

    """
    Args:
        sparkdf: imput edgelist Spark DataFrame
        src: src column name
        dst: dst column name
        distance_colname: distance column name
        cluster_id_colname: Graphframes-created connected components created cluster_id
    Returns:
        node_id:
        eigen_centrality: eigenvector centrality of cluster cluster_id
        cluster_id: cluster_id corresponding to the node_id
Eigenvector Centrality is an algorithm that measures the transitive influence or connectivity of nodes.
Eigenvector Centrality was proposed by Phillip Bonacich, in his 1986 paper Power and Centrality:
A Family of Measures.
It was the first of the centrality measures that considered the transitive importance of a node in a graph,
rather than only considering its direct importance.
Relationships to high-scoring nodes contribute more to the score of a node than connections to low-scoring nodes.
A high score means that a node is connected to other nodes that have high scores.
example input spark dataframe
|src|dst|weight|cluster_id|distance|
|---|---|------|----------|--------|
|  f|  d|  0.67|         0|   0.329|
|  f|  g|  0.34|         0|   0.659|
|  b|  c|  0.56|8589934592|   0.439|
|  g|  h|  0.99|         0|   0.010|
|  a|  b|   0.4|8589934592|     0.6|
|  h|  i|   0.5|         0|     0.5|
|  h|  j|   0.8|         0|   0.199|
|  d|  e|  0.84|         0|   0.160|
|  e|  f|  0.65|         0|    0.35|
example output spark dataframe
|node_id|   eigen_centrality|cluster_id|
|-------|-------------------|----------|
|   b   |  0.707106690085642|8589934592|
|   c   | 0.5000000644180599|8589934592|
|   a   | 0.5000000644180599|8589934592|
|   f   | 0.5746147732828122|         0|
|   d   | 0.4584903903420785|         0|
|   g   |0.37778352393858183|         0|
|   h   |0.27663243805676946|         0|
|   i   |0.12277029263709134|         0|
|   j   |0.12277029263709134|         0|
|   e   | 0.4584903903420785|         0|
    """
    ecschema = StructType(
        [
            StructField("node_id", StringType()),
            StructField("eigen_centrality", DoubleType()),
            StructField(cluster_id_colname, LongType()),
        ]
    )

    psrc = src
    pdst = dst

    @pandas_udf(ecschema, PandasUDFType.GROUPED_MAP)
    def eigenc(pdf: pd.DataFrame) -> pd.DataFrame:
        nxGraph = nx.Graph()
        nxGraph = nx.from_pandas_edgelist(pdf, psrc, pdst)
        ec = eigenvector_centrality(nxGraph, tol=1e-03)
        out_df = (
            pd.DataFrame.from_dict(ec, orient="index", columns=["eigen_centrality"])
            .reset_index()
            .rename(
                columns={"index": "node_id", "eigen_centrality": "eigen_centrality"}
            )
        )

        cluster_id = pdf[cluster_id_colname][0]
        out_df[cluster_id_colname] = cluster_id
        return out_df

    out = sparkdf.groupby(cluster_id_colname).apply(eigenc)
    return out


def centrality(
    sparkdf, centrality_function, src="src", dst="dst", cluster_id_colname="cluster_id", **kwargs,
):

    """
    Args:
        sparkdf: imput edgelist Spark DataFrame
        centrality_function: callable centrality function from networkx
        src: src column name
        dst: dst column name
        distance_colname: distance column name
        cluster_id_colname: Graphframes-created connected components created cluster_id
    Returns:
        node_id:
        <centrality_name>: centrality of cluster cluster_id
        cluster_id: cluster_id corresponding to the node_id

example input spark dataframe
|src|dst|weight|cluster_id|distance|
|---|---|------|----------|--------|
|  f|  d|  0.67|         0|   0.329|
|  f|  g|  0.34|         0|   0.659|
|  b|  c|  0.56|8589934592|   0.439|
|  g|  h|  0.99|         0|   0.010|
|  a|  b|   0.4|8589934592|     0.6|
|  h|  i|   0.5|         0|     0.5|
|  h|  j|   0.8|         0|   0.199|
|  d|  e|  0.84|         0|   0.160|
|  e|  f|  0.65|         0|    0.35|
example output spark dataframe
|node_id|   eigen_centrality|cluster_id|
|-------|-------------------|----------|
|   b   |  0.707106690085642|8589934592|
|   c   | 0.5000000644180599|8589934592|
|   a   | 0.5000000644180599|8589934592|
|   f   | 0.5746147732828122|         0|
|   d   | 0.4584903903420785|         0|
|   g   |0.37778352393858183|         0|
|   h   |0.27663243805676946|         0|
|   i   |0.12277029263709134|         0|
|   j   |0.12277029263709134|         0|
|   e   | 0.4584903903420785|         0|
    """
    centrality_name = centrality_function.__name__
    ecschema = StructType(
        [
            StructField("node_id", StringType()),
            StructField(centrality_name, DoubleType()),
            # StructField(cluster_id_colname, LongType()),
        ]
    )

    psrc = src
    pdst = dst

    @pandas_udf(ecschema, PandasUDFType.GROUPED_MAP)
    def centrality_udf(pdf: pd.DataFrame) -> pd.DataFrame:
        nxGraph = nx.Graph()
        nxGraph = nx.from_pandas_edgelist(pdf, psrc, pdst)
        cent = centrality_function(nxGraph, **kwargs)
        out_df = (
            pd.DataFrame.from_dict(cent, orient="index", columns=[centrality_name])
            .reset_index()
            .rename(
                columns={"index": "node_id", "eigen_centrality": centrality_name}
            )
        )

        # cluster_id = pdf[cluster_id_colname][0]
        # out_df[cluster_id_colname] = cluster_id
        return out_df

    out = sparkdf.apply(centrality_udf)
    return out