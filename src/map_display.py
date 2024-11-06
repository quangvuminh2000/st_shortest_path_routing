import folium
import yaml
import os

# Constraints
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load configuration settings from map_conf.yaml
map_conf_path = os.path.join(SCRIPT_DIR, "../config/map_conf.yaml")
with open(map_conf_path) as conf_file:
    map_conf = yaml.safe_load(conf_file)


# Extract map settings
INITIAL_COORDINATES = map_conf["map_settings"]["initial_coordinates"]
INITIAL_ZOOM = map_conf["map_settings"]["initial_zoom"]
MARKER_COLOR = map_conf["map_settings"]["marker_color"]
INITIAL_ALGORITHM = map_conf["map_settings"]["initial_algorithm"]
START_COORDINATES = map_conf["map_settings"]["start_coordinates"]
END_COORDINATES = map_conf["map_settings"]["end_coordinates"]


def create_map(last_click=None):
    folium_map = folium.Map(location=INITIAL_COORDINATES, zoom_start=INITIAL_ZOOM)

    # Add a click event to the map
    folium.LatLngPopup().add_to(
        folium_map
    )  # This will show the latitude and longitude of clicks

    # If there are last click coordinates, add a marker at that location
    if last_click:
        folium.Marker(
            location=[last_click["lat"], last_click["lng"]],
            popup=f"Clicked Location: {last_click['lat']}, {last_click['lng']}",
            icon=folium.Icon(color="blue"),
        ).add_to(folium_map)

    return folium_map


def add_marker(
    folium_map, location, popup_text="Marker", icon=folium.Icon(color=MARKER_COLOR)
):
    folium.Marker(location, popup=popup_text, icon=icon).add_to(folium_map)
    return folium_map
