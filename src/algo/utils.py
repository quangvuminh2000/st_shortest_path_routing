def get_shortest_path(end, predecessor):  # Dijkstra, Bellman-ford
    path = []
    current_node = end
    while current_node is not None:
        path.append(current_node)
        current_node = predecessor[current_node]
    path.reverse()  # Reverse the path to get the correct order
    return path


def get_path(start, end, predecessor, node_index, index_node):  # Floyd-warshall
    i, j = node_index[start], node_index[end]
    path = []
    while j is not None:
        path.append(j)
        j = predecessor.get((predecessor.get((j, i), None), j), None)
    path.reverse()
    final_path = [index_node[i] for i in path]

    return final_path
