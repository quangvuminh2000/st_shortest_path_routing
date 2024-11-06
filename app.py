import os
import yaml
import warnings

import streamlit as st
import pandas as pd
import folium

from streamlit_folium import st_folium, folium_static
from haversine import haversine, Unit

from networkx import NetworkXNoPath

from src import (
    build_graph,
    create_map,
    INITIAL_COORDINATES,
    INITIAL_ALGORITHM,
    START_COORDINATES,
    END_COORDINATES,
    yen_algorithm,
    # floyd_warshall,
    # bellman_ford,
)
from src.utils import getKNN
from src.algo import (
    dijkstra,
    bellman_ford,
    single_source_bellman_ford,
    floyd_warshall_improved,
    yen_k_shortest_paths,
    k_shortest_paths,
)


warnings.filterwarnings("ignore", category=DeprecationWarning)


# * Page config
st.set_page_config(
    page_title="Group 1: Interactive Shortest Path Finder",
    page_icon=":world_map:",
    layout="wide",
)

COLORS = ["blue", "green", "red", "yellow", "purple"]

# * Starting variables
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load configs
app_conf_path = os.path.join(SCRIPT_DIR, "./config/st_conf.yaml")
with open(app_conf_path) as conf_file:
    app_conf = yaml.safe_load(conf_file)

# Extract app settings
MAP_HEIGHT = app_conf["app_settings"]["map"]["height"]
MAP_WIDTH = app_conf["app_settings"]["map"]["width"]
MAP_SIDEBAR_HEIGHT = app_conf["app_settings"]["map"]["sidebar"]["height"]
MAP_SIDEBAR_WIDTH = app_conf["app_settings"]["map"]["sidebar"]["width"]

# * State variables
if "start_last_clicked" not in st.session_state:
    st.session_state["start_last_clicked"] = INITIAL_COORDINATES

if "source" not in st.session_state:
    st.session_state["source"] = START_COORDINATES

if "end_last_clicked" not in st.session_state:
    st.session_state["end_last_clicked"] = INITIAL_COORDINATES

if "target" not in st.session_state:
    st.session_state["target"] = END_COORDINATES

if "algorithm" not in st.session_state:
    st.session_state["algorithm"] = INITIAL_ALGORITHM

if "k_shortest" not in st.session_state:
    st.session_state["k_shortest"] = 3

# * Helper functions

# * Map
start_map = create_map()
end_map = create_map()

# * Layout
st.title("Welcome to the Shortest Path Finder app!")
st.image(os.path.join(SCRIPT_DIR, "./img/map.jpg"))


# * Sidebar Controls
with st.sidebar:
    st.title("Controls")

    with st.form("input_map_form"):

        # Choose start location
        st.subheader("Choose a starting location")
        st.caption(":blue[Click on the map to choose starting point:]")

        # Create the map to select starting location
        with st.container(key="start_location_map_container"):

            # User pan the maps
            start_map_state_change = st_folium(
                start_map,
                key="start_map",
                height=MAP_SIDEBAR_HEIGHT,
                width=MAP_SIDEBAR_WIDTH,
                returned_objects=["last_clicked"],
            )

            # Check if a click event occurred
            if start_map_state_change and "last_clicked" in start_map_state_change:
                start_last_clicked = start_map_state_change["last_clicked"]
                if start_last_clicked:

                    # Create a new map with the marker at the last clicked location
                    start_map = create_map(start_last_clicked)

                    st.session_state["start_last_clicked"] = [
                        start_map_state_change["last_clicked"]["lat"],
                        start_map_state_change["last_clicked"]["lng"],
                    ]

            else:
                st.write("Click on the map to get coordinates.")

        # Showing the information
        dec = 10
        st.write(
            round(st.session_state["start_last_clicked"][0], dec),
            ", ",
            round(st.session_state["start_last_clicked"][1], dec),
        )

        # Choose end location
        st.subheader("Choose an ending location")
        st.caption(":blue[Click on the map to choose ending point:]")

        # Create the map to select ending location
        with st.container(key="end_location_map_container"):

            # User pan the maps
            end_map_state_change = st_folium(
                end_map,
                key="end_map",
                height=MAP_SIDEBAR_HEIGHT,
                width=MAP_SIDEBAR_WIDTH,
                returned_objects=["last_clicked"],
            )

            if end_map_state_change and "last_clicked" in end_map_state_change:
                end_last_clicked = end_map_state_change["last_clicked"]
                if end_last_clicked:

                    # Create a new map with the marker at the last clicked location
                    end_map = create_map(end_last_clicked)

                    st.session_state["end_last_clicked"] = [
                        end_map_state_change["last_clicked"]["lat"],
                        end_map_state_change["last_clicked"]["lng"],
                    ]

            else:
                st.write("Click on the map to get coordinates.")

        # Showing the information
        dec = 10
        st.write(
            round(st.session_state["end_last_clicked"][0], dec),
            ", ",
            round(st.session_state["end_last_clicked"][1], dec),
        )

        # Choose algorithm
        st.subheader("Choose an algorithm")
        st.caption(":blue[Selection of algorithm for shortest path calculation]")

        # Create the box to select the algorithm
        with st.container(key="algo_selection_container"):
            algorithm = st.selectbox(
                "Choose the algorithm:",
                options=["Dijkstra", "Bellman-Ford", "Floyd-Warshall", "Yen"],
                key="algo_selection_box",
            )
            k_shortest = st.select_slider(
                "Choose the number of shortest paths (Yen's algorithm):",
                help="This setting is only work for Yen's algorithm",
                options=list(range(1, 6)),
                key="k_shortest_select_slider",
            )

        # Submit the selection of settings
        with st.container(key="submit_btn_container"):
            submitted = st.form_submit_button(label="Submit Settings")

            if submitted:
                st.session_state["source"] = st.session_state["start_last_clicked"]
                st.session_state["target"] = st.session_state["end_last_clicked"]
                st.session_state["algorithm"] = algorithm
                st.session_state["k_shortest"] = k_shortest


@st.cache_data
def load_graph_data():
    nodes = pd.read_csv(
        os.path.join(SCRIPT_DIR, "./data/primary_node_list.csv"), index_col=0
    )
    edges = pd.read_csv(os.path.join(SCRIPT_DIR, "./data/primary_edge_list.csv"))
    graph = build_graph(nodes, edges)

    node_dict = nodes[["y", "x"]].to_dict(orient="index")

    return graph, edges, nodes, node_dict


with st.spinner("Loading data..."):
    dir_graph, edges, nodes, node_dict = load_graph_data()

st.success(f"Completely load data of {len(nodes)} nodes and {len(edges)} edges~~")


def run_algorithm(algorithm_name, points, k=3):
    start, end = points[0], points[1]

    nearest_start, nearest_start_loc = getKNN(start, node_dict, nodes)
    nearest_end, nearest_end_loc = getKNN(end, node_dict, nodes)

    distance_start = haversine(start, nearest_start_loc, unit=Unit.METERS)
    distance_end = haversine(end, nearest_end_loc, unit=Unit.METERS)
    print(f"Nearest start, end (meters): {distance_start:4f} {distance_end:4f}")

    if algorithm_name == "Dijkstra":
        path_length, vertices, duration = dijkstra(
            dir_graph, nearest_start, nearest_end
        )
    elif algorithm_name == "Bellman-Ford":
        path_length, vertices, duration = single_source_bellman_ford(
            dir_graph, nearest_start, nearest_end, weight="weight"
        )
    elif algorithm_name == "Floyd-Warshall":
        path_length, vertices, duration = floyd_warshall_improved(
            dir_graph, nearest_start, nearest_end
        )
    elif algorithm_name == "Yen":
        # costs, paths, duration = k_shortest_paths(
        #     dir_graph, nearest_start, nearest_end, k
        # )
        # paths = [
        #     [start]
        #     + [
        #         (float(node_dict[int(node)]["y"]), float(node_dict[(int(node))]["x"]))
        #         for node in path
        #     ]
        #     + [end]
        #     for path in paths
        # ]

        # costs = [distance_start + cost + distance_end for cost in costs]
        # return costs, paths, duration
        paths, duration = yen_algorithm(dir_graph, nearest_start, nearest_end, k)
        paths = [
            [start]
            + [
                (float(node_dict[int(node)]["y"]), float(node_dict[(int(node))]["x"]))
                for node in path
            ]
            + [end]
            for path in paths
        ]

        path_costs = [
            sum(haversine(u, v, unit=Unit.METERS) for (u, v) in zip(path, path[1:]))
            for path in paths
        ]

        costs = [distance_start + cost + distance_end for cost in path_costs]
        return costs, paths, duration
        # cost_paths, duration = yen_k_shortest_paths(
        #     dir_graph, nearest_start, nearest_end, k
        # )

        # costs = [distance_start + cost + distance_end for cost, _ in cost_paths]
        # paths = [
        #     [start]
        #     + [
        #         (float(node_dict[int(node)]["y"]), float(node_dict[(int(node))]["x"]))
        #         for node in path
        #     ]
        #     + [end]
        #     for _, path in cost_paths
        # ]
        # return costs, paths, duration

    coordinates = (
        [start]
        + [
            (float(node_dict[int(node)]["y"]), float(node_dict[(int(node))]["x"]))
            for node in vertices
        ]
        + [end]
    )

    return path_length + distance_start + distance_end, coordinates, duration


# * Map calculation
st.write("Source:", st.session_state["source"])
st.write("Target:", st.session_state["target"])
st.write("Algorithm:", st.session_state["algorithm"])

with st.spinner("Building map and calculate shortest path..."):
    try:
        distance, coordinates, duration = run_algorithm(
            st.session_state["algorithm"],
            [st.session_state["source"], st.session_state["target"]],
            st.session_state["k_shortest"],
        )
        solution_map = create_map()

        # Shortest path line
        if st.session_state["algorithm"] != "Yen":
            km_distance = int(distance) // 1000
            remain_m_distance = distance - km_distance * 1000
            st.write(f"Shortest path: :blue[{km_distance}km {remain_m_distance:.4f}m]")
            st.write(f"Found in: :blue[{duration:.2f}s]")
            folium.PolyLine(
                coordinates,
                color="blue",
                weight=5,
                tooltip=f"Path Length: {distance:.4f} meters",
            ).add_to(solution_map)
            folium.Marker(
                coordinates[0], popup="Start", icon=folium.Icon(color="blue")
            ).add_to(solution_map)
            folium.Marker(
                coordinates[-1], popup="End", icon=folium.Icon(color="red")
            ).add_to(solution_map)
        elif st.session_state["algorithm"] == "Yen":
            for i, dist in enumerate(distance):
                km_dist = int(dist) // 1000
                remain_m_dist = dist - km_dist * 1000
                st.write(
                    f"Shortest path {i+1}th: :blue[{km_dist}km {remain_m_dist:.4f}m]"
                )

            st.write(f"Found in: :blue[{duration:.2f}s]")
            folium.Marker(
                coordinates[0][0], popup="Start", icon=folium.Icon(color="blue")
            ).add_to(solution_map)
            folium.Marker(
                coordinates[0][-1], popup="End", icon=folium.Icon(color="red")
            ).add_to(solution_map)

            for i, coord in enumerate(coordinates):
                folium.PolyLine(
                    coord,
                    color=COLORS[i],
                    weight=5,
                    tooltip=f"Path Length: {distance[i]:.4f} meters",
                ).add_to(solution_map)
        folium_static(solution_map, width=MAP_WIDTH, height=MAP_HEIGHT)
    except NetworkXNoPath as er:
        st.write(":red[Isolated points found, cannot go anywhere]")
        st.image(os.path.join(SCRIPT_DIR, "./img/error.png"))
