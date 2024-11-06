import time
import heapq
from tqdm import tqdm
from .utils import get_shortest_path


def dijkstra(graph, start, end):
    start_time = time.time()
    # Step 1: Initialize distances and previous nodes
    distances = {node: float("inf") for node in graph}
    distances[start] = 0
    priority_queue = [(0, start)]  # (distance, node)
    previous_nodes = {node: None for node in graph}

    # Step 2: Dijkstra algorithm
    with tqdm(
        total=len(graph.nodes), desc="Dijkstra progress", unit="iteration", leave=False
    ) as pbar:
        while priority_queue:
            current_distance, current_node = heapq.heappop(
                priority_queue
            )  # Get the smallest distance with its node in current check

            # Nodes can only be visited once with the shortest distance
            if current_distance > distances[current_node]:
                continue

            # Iterate over all the neighbor nodes of current_node
            for u, v, data in graph.edges(current_node, data=True):
                neighbor = v
                weight = data.get("weight")
                distance = current_distance + weight

                # Only consider this new path if it's better
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(
                        priority_queue, (distance, neighbor)
                    )  # Add (distance, neighbor) to the heap
            pbar.update(1)  # Update the progress bar

    # Get the shortest path from start - end
    shortest_path = get_shortest_path(end, previous_nodes)

    duration = time.time() - start_time
    return distances[end], shortest_path, duration
