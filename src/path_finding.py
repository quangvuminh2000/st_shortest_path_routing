import networkx as nx
import pandas as pd

from haversine import haversine


def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame):
    """Builds a graph from points"""
    G = nx.DiGraph()

    for index, row in nodes.iterrows():
        G.add_node(index, y=row["y"], x=row["x"])
    for _, row in edges.iterrows():
        G.add_edge(row["source"], row["target"], weight=row["length"])

    return G


def dijkstra(graph, start_point, end_point):
    """Calculate the shortest path using Dijkstra's algorithm"""

    length, path = nx.single_source_dijkstra(
        graph, source=start_point, target=end_point
    )

    return length, path


def bellman_ford(graph, start_point, end_point):
    """Calculate the shortest path using Bellman-Ford's algorithm"""

    length, path = nx.single_source_bellman_ford(
        graph, source=start_point, target=end_point
    )

    return length, path


def floyd_warshall(graph, start_point, end_point):
    """Calculate the shortest path using Floyd-Warshall's algorithm"""

    length, path = nx.single_source_bellman_ford(
        graph, source=start_point, target=end_point
    )

    return length, path
