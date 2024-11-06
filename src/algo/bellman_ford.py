import time

from collections import deque

import networkx as nx
import numpy as np
from tqdm import tqdm

from .utils import get_shortest_path, _build_paths_from_predecessors


def bellman_ford(graph, start, end):
    start_time = time.time()
    # Step 1: Initialize distances and predecessors
    distance = {node: np.inf for node in graph.nodes}
    predecessor = {node: None for node in graph.nodes}
    distance[start] = 0
    # Step 2: Relax edges up to |V| - 1 times
    with tqdm(
        total=len(graph.nodes),
        desc="Belman-Ford progress",
        unit="iteration",
        leave=False,
    ) as pbar:
        for _ in range(len(graph.nodes) - 1):
            for u, v, data in graph.edges(data=True):
                weight = data.get("weight")
                if distance[u] != np.inf and distance[u] + weight < distance[v]:
                    distance[v] = distance[u] + weight
                    predecessor[v] = u  # Keep track of the predecessor
            pbar.update(1)

        # Step 3: Check for negative-weight cycles
        for u, v, data in graph.edges(data=True):
            weight = data.get("weight")
            if distance[u] != float("inf") and distance[u] + weight < distance[v]:
                print("Graph contains a negative-weight cycle")
                return None, None

    # Get the shortest path from start - end
    shortest_path = get_shortest_path(end, predecessor)
    duration = time.time() - start_time
    return distance[end], shortest_path, duration


def single_source_bellman_ford(G, source, target, weight="weight"):
    """
    Compute shortest paths and lengths in a weighted graph G

    Parameters
    ----------
    G : NetworkX graph

    source : node label
        Starting node for path

    target : node label, optional
        Ending node for path

    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number.

    Returns
    -------
    distance, path, duration : pair of dictionaries, or numeric and list
        If target is None, returns a tuple of two dictionaries keyed by node.
        The first dictionary stores distance from one of the source nodes.
        The second stores the path from one of the sources to that node.

        If target is not None, returns a tuple of (distance, path) where
        distance is the distance from source to target and path is a list
        representing the path from source to target.

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.
    """
    start = time.time()

    if source == target:
        if source not in G:
            raise nx.NodeNotFound(f"Node {source} is not found in the graph")
        return (0, [source], time.time() - start)

    weight = _weight_function(G, weight)

    paths = {source: [source]}  # dictionary of paths
    dist = _improved_bellman_ford(G, [source], weight, paths=paths, target=target)
    if target is None:
        return (dist, paths, time.time() - start)
    try:
        return (dist[target], paths[target], time.time() - start)
    except KeyError as err:
        msg = f"Node {target} not reachable from {source}"
        raise nx.NetworkXNoPath(
            msg
        ) from err  # Chaining exception for debugging purpose


def _weight_function(G, weight):
    """
    Returns a function that returns the weight of an edge.

    Parameters
    ----------
    G : NetworkX graph.

    weight : string or function
        If it is callable, `weight` itself is returned. If it is a string,
        it is assumed to be the name of the edge attribute that represents
        the weight of an edge. In that case, a function is returned that
        gets the edge weight according to the specified edge attribute.

    Returns
    -------
    function
        This function returns a callable that accepts exactly three inputs:
        a node, an node adjacent to the first one, and the edge attribute
        dictionary for the eedge joining those nodes. That function returns
        a number representing the weight of an edge.
    """
    if callable(weight):
        return weight

    if G.is_multigraph():
        return lambda u, v, d: min(attr.get(weight, 1) for attr in d.values())
    return lambda u, v, data: data.get(weight, 1)


def _inner_bellman_ford(
    G,
    sources,
    weight,
    pred,
    dist=None,
    heuristic=True,
):
    """Inner Relaxation loop for Bellman–Ford algorithm.

    This is an implementation of the SPFA variant.
    See https://en.wikipedia.org/wiki/Shortest_Path_Faster_Algorithm

    Parameters
    ----------
    G : NetworkX graph

    source: list
        List of source nodes. The shortest path from any of the source
        nodes will be found if multiple sources are provided.

    weight : function
        The weight of an edge is the value returned by the function. The
        function must accept exactly three positional arguments: the two
        endpoints of an edge and the dictionary of edge attributes for
        that edge. The function must return a number.

    pred: dict of lists
        dict to store a list of predecessors keyed by that node

    dist: dict, optional (default=None)
        dict to store distance from source to the keyed node
        If None, returned dist dict contents default to 0 for every node in the
        source list

    heuristic : bool
        Determines whether to use a heuristic to early detect negative
        cycles at a hopefully negligible cost.

    Returns
    -------
    node or None
        Return a node `v` where processing discovered a negative cycle.
        If no negative cycle found, return None.

    Raises
    ------
    NodeNotFound
        If any of `source` is not in `G`.
    """
    for s in sources:
        if s not in G:
            raise nx.NodeNotFound(f"Source {s} not in G")

    if pred is None:
        pred = {v: [] for v in sources}

    if dist is None:
        dist = {v: 0 for v in sources}

    # Heuristic Storage setup
    nonexistent_edge = (None, None)
    pred_edge = {v: None for v in sources}
    recent_update = {v: nonexistent_edge for v in sources}

    G_succ = G._adj  # For speed-up (and works for both directed and undirected graphs)
    inf = float("inf")
    n = len(G)

    count = {}
    q = deque(sources)
    in_q = set(sources)
    while q:
        u = q.popleft()
        in_q.remove(u)

        # Skip relaxations if any of the predecessors of u is in the queue.
        if all(pred_u not in in_q for pred_u in pred[u]):
            dist_u = dist[u]
            for v, e in G_succ[u].items():
                dist_v = dist_u + weight(u, v, e)

                if dist_v < dist.get(v, inf):
                    # In this conditional branch we are updating the path with v.
                    # If it happens that some earlier update also added node v
                    # that implies the existence of a negative cycle since
                    # after the update node v would lie on the update path twice.
                    # The update path is stored up to one of the source nodes,
                    # therefore u is always in the dict recent_update
                    if heuristic:
                        if v in recent_update[u]:
                            # Negative cycle found!
                            pred[v].append(u)
                            return v

                        # Transfer the recent update info from u to v if the
                        # same source node is the head of the update path.
                        # If the source node is responsible for the cost update,
                        # then clear the history and use it instead.
                        if v in pred_edge and pred_edge[v] == u:
                            recent_update[v] = recent_update[u]
                        else:
                            recent_update[v] = (u, v)

                    if v not in in_q:
                        q.append(v)
                        in_q.add(v)
                        count_v = count.get(v, 0) + 1
                        if count_v == n:
                            # Negative cycle found!
                            return v

                        count[v] = count_v
                    dist[v] = dist_v
                    pred[v] = [u]
                    pred_edge[v] = u

                elif dist.get(v) is not None and dist_v == dist.get(v):
                    pred[v].append(u)

    # successfully found shortest_path. No negative cycles found.
    return None


def _improved_bellman_ford(
    G,
    source,
    weight,
    pred=None,
    paths=None,
    dist=None,
    target=None,
    heuristic=True,
):
    """
    Calls relaxation loop for improving Bellman-Ford

    This is an implementation of the SPFA variant.
    See https://en.wikipedia.org/wiki/Shortest_Path_Faster_Algorithm

    Parameters
    ----------
    G : NetworkX graph

    source: list
        List of source nodes. The shortest path from any of the source
        nodes will be found if multiple sources are provided.

    weight : function
        The weight of an edge is the value returned by the function. The
        function must accept exactly three positional arguments: the two
        endpoints of an edge and the dictionary of edge attributes for
        that edge. The function must return a number.

    pred: dict of lists, optional (default=None)
        dict to store a list of predecessors keyed by that node
        If None, predecessors are not stored

    paths: dict, optional (default=None)
        dict to store the path list from source to each node, keyed by node
        If None, paths are not stored

    dist: dict, optional (default=None)
        dict to store distance from source to the keyed node
        If None, returned dist dict contents default to 0 for every node in the
        source list

    target: node label, optional
        Ending node for path. Path lengths to other destinations may (and
        probably will) be incorrect.

    heuristic : bool
        Determines whether to use a heuristic to early detect negative
        cycles at a hopefully negligible cost.

    Returns
    -------
    dist : dict
        Returns a dict keyed by node to the distance from the source.
        Dicts for paths and pred are in the mutated input dicts by those names.

    Raises
    ------
    NodeNotFound
        If any of `source` is not in `G`.

    NetworkXUnbounded
        If the (di)graph contains a negative (di)cycle, the
        algorithm raises an exception to indicate the presence of the
        negative (di)cycle.  Note: any negative weight edge in an
        undirected graph is a negative cycle

    """
    if pred is None:
        pred = {v: [] for v in source}

    if dist is None:
        dist = {v: 0 for v in source}

    negative_cycle_found = _inner_bellman_ford(
        G,
        source,
        weight,
        pred,
        dist,
        heuristic,
    )
    if negative_cycle_found is not None:
        raise nx.NetworkXUnbounded("Negative cycle detected.")

    if paths is not None:
        sources = set(source)
        dsts = [target] if target is not None else pred
        for dst in dsts:
            gen = _build_paths_from_predecessors(sources, dst, pred)
            paths[dst] = next(gen)

    return dist
