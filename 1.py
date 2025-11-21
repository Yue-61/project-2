import pandas as pd

bikes_df = pd.read_csv("day_fleet_inventory.csv")

# 看每个小时的总车数
print(bikes_df.groupby("hour")["bikes"].sum())

