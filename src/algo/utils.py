import networkx as nx


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


def _build_paths_from_predecessors(sources, target, pred):
    if target not in pred:
        raise nx.NetworkXNoPath(f"Target {target} cannot be reached from given sources")

    seen = {target}
    stack = [[target, 0]]
    top = 0
    while top >= 0:
        node, i = stack[top]
        if node in sources:
            yield [p for p, n in reversed(stack[: top + 1])]
        if len(pred[node]) > i:
            stack[top][1] = i + 1
            next = pred[node][i]
            if next in seen:
                continue
            else:
                seen.add(next)
            top += 1
            if top == len(stack):
                stack.append([next, 0])
            else:
                stack[top][:] = [next, 0]
        else:
            seen.discard(node)
            top -= 1
