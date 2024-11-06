import numpy as np
import networkx as nx
from tqdm import tqdm
import time

from .utils import get_path


def floyd_warshall_improved(graph, start, end):
    print("Inner Floyd Warshall Progress")
    start_time = time.time()
    node_to_index = {node: i for i, node in enumerate(graph.nodes())}
    index_to_node = {i: node for node, i in node_to_index.items()}

    n = len(graph.nodes)

    # Step 1: Initialize distance and predecessor matrix
    print("\t>>> Initialize distance and predecessor matrix")
    distance = np.full(
        (n, n), np.inf, dtype=float
    )  # Set all initial distance as infinity
    np.fill_diagonal(distance, 0)  # Set the distance between a node and itself as zero

    predecessor = {}

    for u, v, weight in tqdm(
        graph.edges(data=True),
        desc="Initializing progress",
        unit="iteration",
        leave=False,
    ):
        u_index = node_to_index[u]
        v_index = node_to_index[v]
        distance[u_index, v_index] = weight[
            "weight"
        ]  # Fill the actual distance between nodes
        predecessor[(u_index, v_index)] = u_index

    # Step 2: Floyd-Warshall algorithm
    print("\t>>> Floyd-Warshall algorithm")

    # Initialize incoming and outgoing edges count for each node
    in_nodes = np.zeros(n)
    out_nodes = np.zeros(n)

    # Initialize the dictionary containing each node with its incoming and outgoing nodes
    in_list = {v: [] for v in range(n)}
    out_list = {u: [] for u in range(n)}

    # Fill the actual incoming nodes and outgoing nodes
    with tqdm(
        total=n,
        desc="Filling incoming and outgoing nodes",
        unit="iteration",
        leave=False,
    ) as pbar:
        for u in range(n):
            for v in range(n):
                if u != v and distance[u, v] != np.inf:
                    in_nodes[v] += 1
                    out_nodes[u] += 1
                    in_list[v].append(u)
                    out_list[u].append(v)
            pbar.update(1)

    # Initialize the list of nodes to be processed
    all_k = list(range(n))

    # Create a single progress bar for the overall process
    with tqdm(
        total=n, desc="Floyd-Warshall progress", unit="iteration", leave=False
    ) as pbar:
        while all_k:
            # Choose the k with the lowest (incoming nodes + outgoing nodes)
            min_in_x_out = float("inf")
            selected_k = -1
            for k in all_k:
                if (in_nodes[k] + out_nodes[k]) < min_in_x_out:
                    selected_k = k
                    min_in_x_out = in_nodes[k] + out_nodes[k]

            # With this k, run Floyd-Warshall updates
            for i in in_list[selected_k]:
                for j in out_list[selected_k]:
                    if (
                        distance[i, j]
                        > distance[i, selected_k] + distance[selected_k, j]
                    ):
                        if distance[i, j] == np.inf:
                            in_nodes[j] += 1
                            out_nodes[i] += 1
                            in_list[j].append(i)
                            out_list[i].append(j)

                        distance[i, j] = (
                            distance[i, selected_k] + distance[selected_k, j]
                        )
                        predecessor[(i, j)] = predecessor.get((selected_k, j), None)

            # Update the progress bar after processing each selected_k
            pbar.update(1)

            # Remove the selected_k from the list of nodes to be processed
            all_k.remove(selected_k)

    i, j = node_to_index[start], node_to_index[end]
    if distance[i, j] != np.inf:
        duration = time.time() - start_time
        shortest_path = get_path(start, end, predecessor, node_to_index, index_to_node)
        return distance[i, j], shortest_path, duration
    else:
        msg = f"Node {end} not reachable from {start}"
        raise nx.NetworkXNoPath(msg)
