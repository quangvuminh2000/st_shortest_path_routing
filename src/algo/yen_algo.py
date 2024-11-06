import heapq
import time
import networkx as nx
from tqdm import tqdm

from heapq import heappush, heappop
from itertools import count


def yen_k_shortest_paths(G, source, target, K):
    """Yen's K-Shortest Paths Algorithm without warnings for missing nodes."""
    start_time = time.time()

    # Check if source and target nodes exist in the graph
    if source not in G or target not in G:
        return []  # Return an empty list if nodes are not present

    # Find the main shortest path
    main_path = nx.shortest_path(G, source=source, target=target, weight="weight")
    main_cost = nx.shortest_path_length(
        G, source=source, target=target, weight="weight"
    )

    paths = [(main_cost, main_path)]
    candidates = []
    used_nodes = set(main_path)  # Start with the main path nodes as used

    for k in tqdm(range(1, K), desc="Finding K Shortest Paths", leave=False):
        for i in tqdm(
            range(len(paths[k - 1][1]) - 1),
            desc="Exploring nodes...",
            unit="iteration",
            leave=False,
        ):
            spur_node = paths[k - 1][1][i]
            root_path = paths[k - 1][1][: i + 1]

            # Create a deep copy of the graph to modify
            G_spur = G.copy()

            # Remove all nodes used in previous paths except for the start and end nodes
            for node in used_nodes:
                if node != source and node != target and G_spur.has_node(node):
                    G_spur.remove_node(node)

            # Try to find a spur path from the spur node to the target
            try:
                spur_path = nx.shortest_path(G_spur, spur_node, target, weight="weight")
                total_path = root_path[:-1] + spur_path
                total_cost = sum(
                    G[u][v]["weight"] for u, v in zip(total_path[:-1], total_path[1:])
                )
                heapq.heappush(candidates, (total_cost, total_path))
            except nx.NetworkXNoPath:
                continue  # Skip if no valid spur path is found
            except nx.NodeNotFound:
                continue  # Skip missing nodes and continue

        if not candidates:
            break  # No more candidates, stop searching

        # Add the best candidate path to the final list of paths
        cost, path = heapq.heappop(candidates)
        paths.append((cost, path))

        # Mark all nodes in the newly found path as used, excluding start and end nodes
        used_nodes.update(node for node in path if node != source and node != target)

    duration = time.time() - start_time

    return paths, duration


def k_shortest_paths(G, source, target, k=1, weight="weight"):
    """Returns the k-shortest paths from source to target in a weighted graph G.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node

    target : node
       Ending node

    k : integer, optional (default=1)
        The number of shortest paths to find

    weight: string, optional (default='weight')
       Edge data key corresponding to the edge weight

    Returns
    -------
    lengths, paths : lists
       Returns a tuple with two lists.
       The first list stores the length of each k-shortest path.
       The second list stores each k-shortest path.

    Raises
    ------
    NetworkXNoPath
       If no path exists between source and target.

    Examples
    --------
    >>> G=nx.complete_graph(5)
    >>> print(k_shortest_paths(G, 0, 4, 4))
    ([1, 2, 2, 2], [[0, 4], [0, 1, 4], [0, 2, 4], [0, 3, 4]])

    Notes
    ------
    Edge weight attributes must be numerical and non-negative.
    Distances are calculated as sums of weighted edges traversed.

    """
    start_time = time.time()
    if source == target:
        return ([0], [[source]], time.time() - start_time)

    length, path = nx.single_source_dijkstra(G, source, target, weight=weight)
    if target not in path:
        raise nx.NetworkXNoPath("node %s not reachable from %s" % (source, target))

    lengths = [length[target]]
    paths = [path[target]]
    c = count()
    B = []
    G_original = G.copy()

    for i in range(1, k):
        for j in range(len(paths[-1]) - 1):
            spur_node = paths[-1][j]
            root_path = paths[-1][: j + 1]

            edges_removed = []
            for c_path in paths:
                if len(c_path) > j and root_path == c_path[: j + 1]:
                    u = c_path[j]
                    v = c_path[j + 1]
                    if G.has_edge(u, v):
                        edge_attr = G.edge[u][v]
                        G.remove_edge(u, v)
                        edges_removed.append((u, v, edge_attr))

            for n in range(len(root_path) - 1):
                node = root_path[n]
                # out-edges
                for u, v, edge_attr in G.edges_iter(node, data=True):
                    G.remove_edge(u, v)
                    edges_removed.append((u, v, edge_attr))

                if G.is_directed():
                    # in-edges
                    for u, v, edge_attr in G.in_edges_iter(node, data=True):
                        G.remove_edge(u, v)
                        edges_removed.append((u, v, edge_attr))

            spur_path_length, spur_path = nx.single_source_dijkstra(
                G, spur_node, target, weight=weight
            )
            if target in spur_path and spur_path[target]:
                total_path = root_path[:-1] + spur_path[target]
                total_path_length = (
                    get_path_length(G_original, root_path, weight)
                    + spur_path_length[target]
                )
                heappush(B, (total_path_length, next(c), total_path))

            for e in edges_removed:
                u, v, edge_attr = e
                G.add_edge(u, v, edge_attr)

        if B:
            (l, _, p) = heappop(B)
            lengths.append(l)
            paths.append(p)
        else:
            break

    return (lengths, paths, time.time() - start_time)


def get_path_length(G, path, weight="weight"):
    length = 0
    if len(path) > 1:
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]

            length += G.edge[u][v].get(weight, 1)

    return length
