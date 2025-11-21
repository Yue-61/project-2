import folium
import pandas as pd

df = pd.read_csv("selected_stations.csv")

m = folium.Map(location=[55.9533, -3.1883], zoom_start=13)

for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row["center_lat"], row["center_lon"]],
        radius=4,
        color="red",
        fill=True
    ).add_to(m)

m.save("edinburgh_stations.html")
