# ==========================================================
# new.py — Siting + Two-layer Fleet & Rebalancing (no cap & no min-rebalance)
# ==========================================================
import os
import zipfile
from math import radians, cos, sin, asin, sqrt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pulp

# ------------------------------
# 路径设置
# ------------------------------
DATA_DIR = ""    # 如果数据在 ./data 下，就改成 "data"

def data_path(name: str) -> str:
    return os.path.join(DATA_DIR, name)

# ------------------------------
# 全局参数
# ------------------------------
GRID_SIZE_M = 500             # 网格大小 500m
MAX_COVER_DIST_M = 500        # 站点覆盖半径 500m
MAX_NUM_STATIONS = 50         # 必须建 50 个站点

STATION_FIXED_COST = 5323.0   # 每个站点固定成本 (£)
SITING_COST_WEIGHT = 1e-3     # 选址目标中成本惩罚的权重 λ

# Fleet 总车数（用于第 1 层 LP）
TOTAL_BIKES = 500             # 总车数 B^{tot}
STATION_CAPACITY = 26         # 每站最大车辆数 C_g
C_REBAL_PER_KM = 0.18         # 每车·km 重平衡成本 c^{reb} (£)
BIKE_PURCHASE_COST = 913.0    # 每辆车购置成本 (£)

# 第 1 层目标的权重：最大化需求 - 成本
DEMAND_WEIGHT = 1.0           # ω_dem
REBAL_WEIGHT = 0.2            # ω_reb

# 定价模型参数（£/hour）
P_REF = 6.6
ALPHA_PRICE_ELASTICITY = 1.2
USER_PRICE_GRID = np.arange(4.26, 5.12, 0.1)
EMPLOYER_SUBSIDY_GRID = np.arange(0.0, 4, 0.1)


# ==========================================================
# 1. 读取数据
# ==========================================================
station_df = pd.read_csv(data_path("station_data.csv"))
pois_df = pd.read_csv(data_path("edinburgh_pois.csv"))

print("Stations:", station_df.shape)
print("POIs:", pois_df.shape)


# ==========================================================
# 2. 地理工具 & 网格
# ==========================================================
def haversine(lat1, lon1, lat2, lon2):
    """计算两点间球面距离（米）"""
    R = 6371000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon/2) ** 2
    c = 2 * asin(sqrt(a))
    return R * c


def build_grid(bounds, cell_size_m=GRID_SIZE_M):
    min_lat, max_lat, min_lon, max_lon = bounds
    lat0 = (min_lat + max_lat) / 2
    meters_per_deg_lat = 111_000
    meters_per_deg_lon = 111_000 * cos(radians(lat0))

    dlat = cell_size_m / meters_per_deg_lat
    dlon = cell_size_m / meters_per_deg_lon

    lat_edges = np.arange(min_lat, max_lat + dlat, dlat)
    lon_edges = np.arange(min_lon, max_lon + dlon, dlon)

    cells = []
    grid_id = 0
    for i in range(len(lat_edges) - 1):
        for j in range(len(lon_edges) - 1):
            center_lat = (lat_edges[i] + lat_edges[i+1]) / 2
            center_lon = (lon_edges[j] + lon_edges[j+1]) / 2
            cells.append({
                "grid_id": grid_id,
                "i": i,
                "j": j,
                "center_lat": center_lat,
                "center_lon": center_lon
            })
            grid_id += 1

    grid_df = pd.DataFrame(cells)
    return grid_df, lat_edges, lon_edges


def assign_points_to_grid(df, lat_col, lon_col, lat_edges, lon_edges):
    df = df.copy()
    df["i"] = np.searchsorted(lat_edges, df[lat_col], side="right") - 1
    df["j"] = np.searchsorted(lon_edges, df[lon_col], side="right") - 1

    mask = (df["i"] >= 0) & (df["j"] >= 0) & \
           (df["i"] < len(lat_edges) - 1) & (df["j"] < len(lon_edges) - 1)
    df = df[mask]

    n_lon_cells = len(lon_edges) - 1
    df["grid_id"] = df["i"] * n_lon_cells + df["j"]
    return df


# ==========================================================
# 3. 网格 + POI 映射
# ==========================================================
all_lats = pd.concat([station_df["lat"], pois_df["lat"]])
all_lons = pd.concat([station_df["lon"], pois_df["lon"]])

lat_margin = 0.005
lon_margin = 0.01

min_lat = all_lats.min() - lat_margin
max_lat = all_lats.max() + lat_margin
min_lon = all_lons.min() - lon_margin
max_lon = all_lons.max() + lon_margin

bounds = (min_lat, max_lat, min_lon, max_lon)
grid_df, lat_edges, lon_edges = build_grid(bounds)

stations_grid = assign_points_to_grid(station_df, "lat", "lon", lat_edges, lon_edges)
pois_grid = assign_points_to_grid(pois_df, "lat", "lon", lat_edges, lon_edges)

poi_counts = pois_grid.groupby(["grid_id", "category"]).size().unstack(fill_value=0)
poi_counts.columns = [f"poi_{c}" for c in poi_counts.columns]

print("Number of grids:", grid_df.shape[0])
print("POI columns:", list(poi_counts.columns))


# ==========================================================
# 4. counts-data → 基础需求
# ==========================================================
def load_all_counts(zip_path):
    all_counts = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        for name in z.namelist():
            if name.endswith("_counts.csv"):
                with z.open(name) as f:
                    df = pd.read_csv(f)
                    all_counts.append(df)
    if not all_counts:
        raise RuntimeError("No *_counts.csv found.")
    return pd.concat(all_counts, ignore_index=True)

counts_all = load_all_counts(data_path("counts-data.zip"))
print("Counts shape:", counts_all.shape)

station_demand = counts_all.groupby("start_station_id")["trip_count"].sum().reset_index()
station_demand.rename(columns={"trip_count": "station_trip_total"}, inplace=True)

station_with_dem = station_df.merge(
    station_demand,
    left_on="station_id",
    right_on="start_station_id",
    how="left"
)
station_with_dem["station_trip_total"] = station_with_dem["station_trip_total"].fillna(0)

station_with_dem_grid = assign_points_to_grid(
    station_with_dem, "lat", "lon", lat_edges, lon_edges
)

grid_base_demand = station_with_dem_grid.groupby("grid_id")["station_trip_total"].sum()
grid_base_demand = grid_base_demand.rename("d_base")

print("Non-zero base demand grids:", (grid_base_demand > 0).sum())


# ==========================================================
# 5. Demand model：POI 回归
# ==========================================================
demand_reg_df = pd.DataFrame(grid_base_demand).join(poi_counts, how="left").fillna(0)
X = demand_reg_df.drop(columns=["d_base"])
y = demand_reg_df["d_base"].values

if X.shape[1] > 0 and y.sum() > 0:
    reg = LinearRegression()
    reg.fit(X, y)
    betas = reg.coef_
    beta_abs_sum = np.sum(np.abs(betas))
    if beta_abs_sum == 0:
        w = np.zeros_like(betas)
    else:
        w = betas / (1 + beta_abs_sum)
    weights = pd.Series(w, index=X.columns, name="w_p")
    print("POI 权重 w_p:")
    print(weights)

    full_X = poi_counts.reindex(grid_df["grid_id"]).fillna(0)
    d_hat = reg.predict(full_X)
    d_hat = np.maximum(d_hat, 0)
    grid_df["demand"] = d_hat
else:
    print("回归数据不足，使用 d_base。")
    grid_df = grid_df.merge(grid_base_demand, on="grid_id", how="left")
    grid_df["demand"] = grid_df["d_base"].fillna(0)

print(grid_df[["grid_id", "center_lat", "center_lon", "demand"]].head())


# ==========================================================
# 6. 小时模式（供 fleet 使用）
# ==========================================================
hourly_totals = counts_all.groupby("hour")["trip_count"].sum()
hourly_factor = (hourly_totals / hourly_totals.mean()).sort_index()
print("Hourly factors:\n", hourly_factor)

plt.figure(figsize=(8, 3))
plt.bar(hourly_factor.index, hourly_factor.values)
plt.xlabel("Hour")
plt.ylabel("Factor")
plt.title("Hourly pattern (relative)")
plt.show()


# ==========================================================
# 7. 基本可视化
# ==========================================================
plt.figure(figsize=(8, 8))
plt.scatter(
    grid_df["center_lon"], grid_df["center_lat"],
    c=grid_df["demand"], s=20, alpha=0.7
)
plt.colorbar(label="Demand")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Grid-level demand")
plt.gca().set_aspect("equal", "box")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(station_df["lon"], station_df["lat"], s=10, alpha=0.7)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Existing stations")
plt.gca().set_aspect("equal", "box")
plt.grid(True)
plt.show()


# ==========================================================
# 8. Siting model：MCLP + 成本惩罚，站点数 = 50
# ==========================================================
cand_grids = grid_df[grid_df["demand"] > 0].copy()
cand_ids = list(cand_grids["grid_id"])
print("Candidate grids:", len(cand_ids))

cent_lat = cand_grids.set_index("grid_id")["center_lat"].to_dict()
cent_lon = cand_grids.set_index("grid_id")["center_lon"].to_dict()
d_dict = cand_grids.set_index("grid_id")["demand"].to_dict()

A = {}
for g in cand_ids:
    for h in cand_ids:
        dist = haversine(cent_lat[g], cent_lon[g], cent_lat[h], cent_lon[h])
        A[(g, h)] = 1 if dist <= MAX_COVER_DIST_M else 0

m_siting = pulp.LpProblem("Siting_MCLP_CostPenalty", pulp.LpMaximize)

x = pulp.LpVariable.dicts("x", cand_ids, lowBound=0, upBound=1, cat="Binary")
cov = pulp.LpVariable.dicts("cov", cand_ids, lowBound=0, upBound=1, cat="Binary")

m_siting += (
    pulp.lpSum(d_dict[g] * cov[g] for g in cand_ids)
    - SITING_COST_WEIGHT * STATION_FIXED_COST * pulp.lpSum(x[g] for g in cand_ids)
)

for g in cand_ids:
    m_siting += pulp.lpSum(A[(h, g)] * x[h] for h in cand_ids) >= cov[g]

m_siting += pulp.lpSum(x[g] for g in cand_ids) == MAX_NUM_STATIONS

print("Solving siting model...")
m_siting.solve(pulp.PULP_CBC_CMD(msg=1))
print("Siting status:", pulp.LpStatus[m_siting.status])

selected_ids = [g for g in cand_ids if pulp.value(x[g]) > 0.5]
print("Selected stations:", len(selected_ids))

selected_df = cand_grids[cand_grids["grid_id"].isin(selected_ids)].copy()
selected_df = selected_df[["grid_id", "center_lat", "center_lon", "demand"]]
selected_df = selected_df.sort_values("demand", ascending=False).reset_index(drop=True)
selected_df.to_csv("selected_stations.csv", index=False)

plt.figure(figsize=(8, 8))
plt.scatter(
    grid_df["center_lon"], grid_df["center_lat"],
    c=grid_df["demand"], s=10, alpha=0.5, label="Demand"
)
plt.colorbar(label="Demand")
plt.scatter(
    selected_df["center_lon"], selected_df["center_lat"],
    s=50, marker="^", edgecolors="k", label="New stations"
)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("New station siting (50 sites)")
plt.legend()
plt.gca().set_aspect("equal", "box")
plt.grid(True)
plt.show()


# ==========================================================
# 9. 第 1 层：白天 Fleet & Rebalancing LP（去掉约束5、6）
# ==========================================================
K_NEIGHBORS = 3  # 每站只和最近 K 个邻居之间允许重平衡


def solve_day_fleet_model(selected_df,
                          hourly_factor,
                          total_bikes=TOTAL_BIKES,
                          station_capacity=STATION_CAPACITY,
                          c_reb_per_km=C_REBAL_PER_KM,
                          demand_weight=DEMAND_WEIGHT,
                          reb_weight=REBAL_WEIGHT,
                          k_neighbors=K_NEIGHBORS):
    """
    白天 fleet & inventory 模型：
      - 决策 B_g^t, q_g^t, r_{g,g'}^t
      - 已知总车数 total_bikes
      - 目标：max ω_dem * Σ q - ω_reb * c_reb * Σ dist * r
      - 不再包括：
          * 每小时重平衡能力限制
          * Σ r >= ε 的非零重平衡约束
    """
    S = list(selected_df["grid_id"])
    T = list(hourly_factor.index.astype(int))
    T.sort()
    t0 = T[0]

    # 1) 构造小时需求 D_{g,t}
    daily_demand = selected_df.set_index("grid_id")["demand"].to_dict()
    hf_norm = hourly_factor / hourly_factor.sum()
    D = {(g, t): daily_demand[g] * hf_norm.loc[t] for g in S for t in T}

    # 2) 距离矩阵与最近邻
    lat_map = selected_df.set_index("grid_id")["center_lat"].to_dict()
    lon_map = selected_df.set_index("grid_id")["center_lon"].to_dict()

    dist_km = {}
    for g in S:
        for gp in S:
            if g == gp:
                dist_km[(g, gp)] = 0.0
            else:
                dist_km[(g, gp)] = haversine(
                    lat_map[g], lon_map[g],
                    lat_map[gp], lon_map[gp]
                ) / 1000.0

    neighbors = {g: [] for g in S}
    for g in S:
        others = [gp for gp in S if gp != g]
        others_sorted = sorted(others, key=lambda gp: dist_km[(g, gp)])
        neighbors[g] = others_sorted[:k_neighbors]

    # 3) 建模
    m = pulp.LpProblem("Day_Fleet_Rebalancing_NoCaps", pulp.LpMaximize)

    B = pulp.LpVariable.dicts("B", (S, T), lowBound=0)
    q = pulp.LpVariable.dicts("q", (S, T), lowBound=0)
    r = {}
    for g in S:
        for gp in neighbors[g]:
            for t in T:
                r[(g, gp, t)] = pulp.LpVariable(f"r_{g}_{gp}_{t}", lowBound=0)

    # 目标函数
    total_served = pulp.lpSum(q[g][t] for g in S for t in T)
    total_reb_cost = c_reb_per_km * pulp.lpSum(
        dist_km[(g, gp)] * r[(g, gp, t)] for (g, gp, t) in r
    )
    m += demand_weight * total_served - reb_weight * total_reb_cost

    # 需求上界
    for g in S:
        for t in T:
            m += q[g][t] <= D[(g, t)]

    # 库存动态 & 容量
    for g in S:
        for i, t in enumerate(T[:-1]):
            t_next = T[i + 1]
            inbound = pulp.lpSum(
                r[(gp, g, t)] for gp in S if (gp, g, t) in r
            )
            outbound = pulp.lpSum(
                r[(g, gp, t)] for gp in neighbors[g]
            )
            m += B[g][t_next] == B[g][t] - q[g][t] + inbound - outbound
    for g in S:
        for t in T:
            m += B[g][t] <= station_capacity

    # 初始总车数
    m += pulp.lpSum(B[g][t0] for g in S) == total_bikes

    print("Solving day fleet model (no rebalance caps)...")
    m.solve(pulp.PULP_CBC_CMD(msg=1))
    print("Day fleet status:", pulp.LpStatus[m.status])

    B_sol = {(g, t): pulp.value(B[g][t]) for g in S for t in T}
    q_sol = {(g, t): pulp.value(q[g][t]) for g in S for t in T}
    r_sol = {(g, gp, t): pulp.value(var) for (g, gp, t), var in r.items()}

    total_served_value = sum(q_sol[(g, t)] for g in S for t in T)
    total_reb_moves = sum((r_sol[(g, gp, t)] or 0.0) for (g, gp, t) in r_sol)
    rebalance_cost_day = c_reb_per_km * sum(
        dist_km[(g, gp)] * (r_sol[(g, gp, t)] or 0.0)
        for (g, gp, t) in r_sol
    )

    print("Day served demand:", total_served_value)
    print("Day rebalancing moves:", total_reb_moves)
    print("Day rebalancing cost (£):", rebalance_cost_day)

    bikes_records = []
    for g in S:
        for t in T:
            bikes_records.append(
                {"grid_id": g, "hour": t, "bikes": B_sol[(g, t)]}
            )
    bikes_df = pd.DataFrame(bikes_records)
    bikes_df = bikes_df.merge(
        selected_df[["grid_id", "center_lat", "center_lon"]],
        on="grid_id", how="left"
    )
    bikes_df.to_csv("day_fleet_inventory.csv", index=False)

    rebal_records = []
    for (g, gp, t), val in r_sol.items():
        if val is not None and val > 1e-6:
            rebal_records.append(
                {"from": g, "to": gp, "hour": t, "bikes_moved": val}
            )
    rebal_df = pd.DataFrame(rebal_records)
    rebal_df.to_csv("day_rebalancing_flows.csv", index=False)

    return m, bikes_df, rebal_df, total_served_value, rebalance_cost_day


m_day_fleet, bikes_df, day_rebal_df, served_total, rebalance_cost_day = \
    solve_day_fleet_model(selected_df, hourly_factor)

print("Day fleet model solved.")


# ==========================================================
# 10. 第 2 层：夜间静态重平衡 = 最小费用流 LP
# ==========================================================
def solve_overnight_rebalance_min_cost_flow(selected_df,
                                            bikes_df,
                                            c_reb_per_km=C_REBAL_PER_KM):
    """
    输入：
      selected_df: grid_id, center_lat, center_lon
      bikes_df   : grid_id, hour, bikes （来自白天 LP）
    用简单的最小费用流近似 nightly rebalancing。
    """
    S = list(selected_df["grid_id"])

    # 1) 当日结束库存 B_end[g]（用最后一个 hour）
    last_hour = bikes_df["hour"].max()
    B_end_series = bikes_df[bikes_df["hour"] == last_hour].set_index("grid_id")["bikes"]
    B_end = {g: B_end_series.get(g, 0.0) for g in S}

    # 2) 目标库存 B_target[g]（这里用全天平均库存，截断到 [0,C]）
    B_avg_series = bikes_df.groupby("grid_id")["bikes"].mean()
    B_target = {}
    for g in S:
        avg = B_avg_series.get(g, 0.0)
        B_target[g] = max(0.0, min(STATION_CAPACITY, avg))

    # 3) 需求 d_g = B_target - B_end
    d = {g: B_target[g] - B_end[g] for g in S}
    total_d = sum(d.values())

    # 如果总和不是 0，加入虚拟 depot 补齐
    use_depot = abs(total_d) > 1e-6
    depot = "DEPOT"
    if use_depot:
        S_ext = S + [depot]
        d_ext = dict(d)
        d_ext[depot] = -total_d
    else:
        S_ext = S
        d_ext = d

    # 4) 距离
    lat_map = selected_df.set_index("grid_id")["center_lat"].to_dict()
    lon_map = selected_df.set_index("grid_id")["center_lon"].to_dict()

    def get_coord(node):
        if node == depot:
            return (np.mean(list(lat_map.values())), np.mean(list(lon_map.values())))
        else:
            return lat_map[node], lon_map[node]

    dist_km = {}
    for i in S_ext:
        for j in S_ext:
            if i == j:
                dist_km[(i, j)] = 0.0
            else:
                lat1, lon1 = get_coord(i)
                lat2, lon2 = get_coord(j)
                dist_km[(i, j)] = haversine(lat1, lon1, lat2, lon2) / 1000.0

    # 5) 最小费用流 LP
    m = pulp.LpProblem("Overnight_Rebalancing_MinCostFlow", pulp.LpMinimize)
    f = pulp.LpVariable.dicts("f", (S_ext, S_ext), lowBound=0)

    m += c_reb_per_km * pulp.lpSum(
        dist_km[(i, j)] * f[i][j] for i in S_ext for j in S_ext
    )

    for i in S_ext:
        m += (
            pulp.lpSum(f[i][j] for j in S_ext)
            - pulp.lpSum(f[j][i] for j in S_ext)
            == d_ext.get(i, 0.0)
        )

    print("Solving overnight min-cost flow...")
    m.solve(pulp.PULP_CBC_CMD(msg=1))
    print("Overnight status:", pulp.LpStatus[m.status])

    records = []
    total_cost = 0.0
    for i in S_ext:
        for j in S_ext:
            if i == j:
                continue
            val = pulp.value(f[i][j])
            if val is not None and val > 1e-6:
                records.append({"from": i, "to": j, "bikes_moved": val})
                total_cost += c_reb_per_km * dist_km[(i, j)] * val

    flows_df = pd.DataFrame(records)
    flows_df.to_csv("overnight_rebalancing_flows.csv", index=False)
    print("Overnight rebalancing cost (£):", total_cost)
    return flows_df, total_cost


overnight_flows_df, rebalance_cost_night = \
    solve_overnight_rebalance_min_cost_flow(selected_df, bikes_df)

total_rebalance_cost = rebalance_cost_day + rebalance_cost_night
print("Total rebalancing cost (day + night) (£):", total_rebalance_cost)


# ========================================================== 
# 11. Price model —— 最大化全年利润，对应论文中的模型
# ==========================================================

# --- 成本部分（按“年”口径），对应 C^site, C^bike, C^reb,year, C^year ---

num_stations = len(selected_ids)

C_site = num_stations * STATION_FIXED_COST          # 站点年成本 C^site
C_bike = TOTAL_BIKES * BIKE_PURCHASE_COST           # 车队年成本 C^bike
C_reb_year = total_rebalance_cost * 365.0           # 重平衡年成本 C^reb,year
C_year = C_site + C_bike + C_reb_year               # 系统年总成本 C^year

print("Annual siting cost      C_site      (£/year):", C_site)
print("Annual bike cost        C_bike      (£/year):", C_bike)
print("Annual rebalancing cost C_reb,year  (£/year):", C_reb_year)
print("Total system cost       C_year      (£/year):", C_year)

# --- 基础日需求 d0：由 fleet & rebalancing 模型得到的“容量约束下每日可服务需求” ---

d0 = served_total  # trips/day
print("Capacity-limited base demand d0 (trips/day):", d0)

# 方便对比：理论需求（不考虑容量约束）
theoretical_demand_daily = grid_df["demand"].sum()
print("Theoretical demand at p_ref=6.6 (no capacity limit, trips/day):",
      theoretical_demand_daily)


def pricing_model_max_annual_profit(
    d0,
    C_year,
    user_price_grid=USER_PRICE_GRID,
    employer_subsidy_grid=EMPLOYER_SUBSIDY_GRID,
    alpha=ALPHA_PRICE_ELASTICITY,
    p_ref=P_REF,
):
    """
    对应论文中的 price model：

      D_day(p_u, s_e) = d0 * max{ 0, 1 - α (p_net - p_ref) / p_ref }
      R_year          = 365 (p_u + s_e) D_day
      Π_year          = R_year - C_year

    在离散网格 (p_u, s_e) 上枚举，找到 Π_year 最大的组合。
    """

    best = {
        "p_u": None,
        "s_e": None,
        "D_day": 0.0,
        "R_day": 0.0,
        "R_year": 0.0,
        "Pi_year": -1e18,
    }

    p_min = float(min(user_price_grid))
    p_max = float(max(user_price_grid))
    s_e_max = float(max(employer_subsidy_grid))
    print(f"\nPricing search grid: p_u ∈ [{p_min}, {p_max}], s_e ∈ [0, {s_e_max}]")

    for p_u in user_price_grid:
        for s_e in employer_subsidy_grid:

            # (2) 净价格定义与非负性：p_net = max{p_u - s_e, 0}
            p_net = max(p_u - s_e, 0.0)

            # 价格弹性需求因子：1 - α (p_net - p_ref) / p_ref
            demand_factor = 1.0 - alpha * (p_net - p_ref) / p_ref

            # (3) 需求非负性：D_day ≥ 0
            demand_factor = max(demand_factor, 0.0)

            # 需求函数：D_day(p_u, s_e)
            D_day = d0 * demand_factor

            # 每日收入与全年收入
            R_day = (p_u + s_e) * D_day
            R_year = 365.0 * R_day

            # 年利润 Π_year
            Pi_year = R_year - C_year

            # 目标：max Π_year
            if Pi_year > best["Pi_year"]:
                best = {
                    "p_u": p_u,
                    "s_e": s_e,
                    "D_day": D_day,
                    "R_day": R_day,
                    "R_year": R_year,
                    "Pi_year": Pi_year,
                }

    return best


# 用上面的模型求最优 (p_u, s_e)
best_price = pricing_model_max_annual_profit(d0, C_year)

print("\n======== 最佳定价方案（以全年利润为目标） ========")
print("  用户票价 p_u           = %.2f £/hour" % best_price["p_u"])
print("  雇主补贴 s_e           = %.2f £/hour" % best_price["s_e"])
print("  每日需求 D_day         = %.1f trips/day" % best_price["D_day"])
print("  每日收入 R_day         = %.1f £/day" % best_price["R_day"])
print("  年收入  R_year         = %.1f £/year" % best_price["R_year"])
print("  年成本  C_year         = %.1f £/year" % C_year)
print("  年利润  Π_year         = %.1f £/year" % best_price["Pi_year"])
print("=================================================\n")

daily_revenue = best_price["R_day"]
monthly_revenue = daily_revenue * 30.0
yearly_revenue = best_price["R_year"]

print("==============================")
print(" 简单收入评估（不含季节变化）")
print("==============================")
print("每天收入 R_day          : %.2f £/day" % daily_revenue)
print("每月收入 (按 30 天)     : %.2f £/month" % monthly_revenue)
print("全年收入 R_year         : %.2f £/year" % yearly_revenue)
print("对应年利润 Π_year       : %.2f £/year" % best_price["Pi_year"])
print("==============================\n")


