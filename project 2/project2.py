import pandas as pd
import numpy as np
import zipfile
import os
from glob import glob
import pulp
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class EdinburghBikeDataLoader:
    def __init__(self, data_folder="."):
        self.data_folder = data_folder
    
    def load_station_data(self):
        """读取站点数据"""
        try:
            stations = pd.read_csv('station_data.csv')
            print(f"成功加载站点数据，共 {len(stations)} 个站点")
            # 清理数据
            stations = stations.dropna(subset=['lat', 'lon'])
            return stations
        except FileNotFoundError:
            print("未找到 station_data.csv 文件")
            return None
    
    def load_poi_data(self):
        """读取POI数据"""
        try:
            pois = pd.read_csv('edinburgh_pois.csv')
            print(f"成功加载POI数据，共 {len(pois)} 个POI")
            # 清理数据
            pois = pois.dropna(subset=['lat', 'lon'])
            return pois
        except FileNotFoundError:
            print("未找到 edinburgh_pois.csv 文件")
            return None
    
    def load_cycle_hire_data(self):
        """读取骑行数据（从zip文件中）"""
        try:
            all_trips = []
            
            with zipfile.ZipFile('cyclehire-data.zip', 'r') as zip_ref:
                csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                
                for csv_file in csv_files:
                    print(f"正在读取: {csv_file}")
                    with zip_ref.open(csv_file) as f:
                        df = pd.read_csv(f)
                        all_trips.append(df)
            
            if all_trips:
                combined_trips = pd.concat(all_trips, ignore_index=True)
                print(f"成功加载骑行数据，共 {len(combined_trips)} 条记录")
                return combined_trips
            else:
                print("未在zip文件中找到任何CSV文件")
                return None
        except FileNotFoundError:
            print("未找到 cyclehire-data.zip 文件")
            return None
    
    def load_counts_data(self):
        """读取计数数据（从zip文件中）"""
        try:
            all_counts = []
            
            with zipfile.ZipFile('counts-data.zip', 'r') as zip_ref:
                csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                
                for csv_file in csv_files:
                    print(f"正在读取: {csv_file}")
                    with zip_ref.open(csv_file) as f:
                        df = pd.read_csv(f)
                        all_counts.append(df)
            
            if all_counts:
                combined_counts = pd.concat(all_counts, ignore_index=True)
                print(f"成功加载计数数据，共 {len(combined_counts)} 条记录")
                return combined_counts
            else:
                print("未在zip文件中找到任何CSV文件")
                return None
        except FileNotFoundError:
            print("未找到 counts-data.zip 文件")
            return None

class BikeSharingModel:
    def __init__(self, grid_size=500):
        self.grid_size = grid_size
        self.grids = None
        self.demand_model = None
        self.site_selection_model = None
        
    def create_grid_system(self, bounds):
        """
        创建网格系统
        bounds: (min_lon, max_lon, min_lat, max_lat)
        """
        min_lon, max_lon, min_lat, max_lat = bounds
        
        # 计算网格数量
        lon_steps = int((max_lon - min_lon) * 111320 / self.grid_size)
        lat_steps = int((max_lat - min_lat) * 111320 / self.grid_size)
        
        grids = []
        grid_id = 0
        
        for i in range(lon_steps):
            for j in range(lat_steps):
                center_lon = min_lon + (i + 0.5) * (self.grid_size / 111320)
                center_lat = min_lat + (j + 0.5) * (self.grid_size / 111320)
                
                grid = {
                    'grid_id': grid_id,
                    'center_lon': center_lon,
                    'center_lat': center_lat,
                    'bounds': (
                        min_lon + i * (self.grid_size / 111320),
                        min_lon + (i + 1) * (self.grid_size / 111320),
                        min_lat + j * (self.grid_size / 111320),
                        min_lat + (j + 1) * (self.grid_size / 111320)
                    )
                }
                grids.append(grid)
                grid_id += 1
        
        self.grids = pd.DataFrame(grids)
        print(f"创建了 {len(self.grids)} 个网格")
        return self.grids
    
    def assign_locations_to_grids(self, locations_df, lon_col, lat_col, keep_columns=None):
        """将位置数据分配到网格 - 修复版本"""
        assignments = []
        
        for idx, location in locations_df.iterrows():
            lon, lat = location[lon_col], location[lat_col]
            
            for _, grid in self.grids.iterrows():
                min_lon, max_lon, min_lat, max_lat = grid['bounds']
                if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
                    assignment = {
                        'location_id': idx,
                        'grid_id': grid['grid_id'],
                        'lon': lon,
                        'lat': lat
                    }
                    
                    # 保留指定的列
                    if keep_columns:
                        for col in keep_columns:
                            if col in location:
                                assignment[col] = location[col]
                    
                    assignments.append(assignment)
                    break
        
        return pd.DataFrame(assignments)

class DemandModel:
    def __init__(self, grids):
        self.grids = grids
        self.poi_weights = None
        self.time_factors = None
        
    def calculate_base_demand(self, historical_trips):
        """计算基础需求"""
        if 'start_station_id' not in historical_trips.columns or 'trip_count' not in historical_trips.columns:
            print("历史行程数据格式不正确")
            return None
            
        base_demand = historical_trips.groupby('start_station_id')['trip_count'].mean()
        return base_demand
    
    def calculate_poi_weights(self, poi_data, trip_data):
        """使用回归分析计算POI权重 - 修复版本"""
        try:
            # 检查POI数据是否包含类别信息
            if 'category' not in poi_data.columns:
                print("POI数据缺少类别信息，使用默认权重")
                # 返回默认权重
                default_categories = ['residential', 'commercial', 'school', 'library', 'university', 'hospital']
                default_weights = {cat: 0.1 for cat in default_categories}
                self.poi_weights = default_weights
                return default_weights
            
            # 准备数据
            X = []
            y = []
            
            # 获取所有POI类别
            poi_categories = poi_data['category'].unique()
            print(f"POI类别: {poi_categories}")
            
            # 使用网格采样以减少计算量
            if len(self.grids) > 200:
                sample_grids = self.grids.sample(n=200, random_state=42)
                print(f"使用 {len(sample_grids)} 个网格样本进行计算")
            else:
                sample_grids = self.grids
            
            for _, grid in sample_grids.iterrows():
                grid_id = grid['grid_id']
                grid_poi = poi_data[poi_data['grid_id'] == grid_id]
                
                # POI数量特征
                poi_features = []
                for poi_type in poi_categories:
                    count = len(grid_poi[grid_poi['category'] == poi_type])
                    poi_features.append(count)
                
                # 需求目标变量 - 使用该网格内所有站点的需求总和
                grid_station_ids = poi_data[
                    (poi_data['grid_id'] == grid_id) & 
                    (poi_data['location_id'].isin(trip_data['start_station_id'].unique()))
                ]['location_id'].unique()
                
                if len(grid_station_ids) > 0:
                    demand = trip_data[trip_data['start_station_id'].isin(grid_station_ids)]['trip_count'].sum()
                else:
                    demand = 0
                
                X.append(poi_features)
                y.append(demand)
            
            X = np.array(X)
            y = np.array(y)
            
            # 检查数据有效性
            if np.sum(y) == 0:
                print("警告：所有需求值都为0，使用默认权重")
                default_weights = {category: 0.1 for category in poi_categories}
                self.poi_weights = default_weights
                return default_weights
            
            # 线性回归
            model = LinearRegression()
            model.fit(X, y)
            
            # 标准化权重
            coefficients = model.coef_
            total_coef = np.sum(np.abs(coefficients)) if np.sum(np.abs(coefficients)) > 0 else 1
            weights_array = coefficients / (1 + total_coef)
            
            # 确保poi_weights是字典格式
            self.poi_weights = dict(zip(poi_categories, weights_array))
            
            print("POI权重计算完成")
            return self.poi_weights
        except Exception as e:
            print(f"计算POI权重时出错: {e}")
            # 返回默认权重
            if 'category' in poi_data.columns:
                default_weights = {category: 0.1 for category in poi_data['category'].unique()}
            else:
                default_categories = ['residential', 'commercial', 'school', 'library', 'university', 'hospital']
                default_weights = {cat: 0.1 for cat in default_categories}
            self.poi_weights = default_weights
            return default_weights
    
    def calculate_time_factors(self, hourly_trips):
        """计算时间变化因子"""
        if 'hour' not in hourly_trips.columns or 'trip_count' not in hourly_trips.columns:
            print("小时行程数据格式不正确")
            return None
            
        total_trips = hourly_trips['trip_count'].sum()
        if total_trips > 0:
            self.time_factors = hourly_trips.groupby('hour')['trip_count'].sum() / total_trips
        else:
            self.time_factors = pd.Series([1/24] * 24, index=range(24))
        
        return self.time_factors
    
    def predict_demand(self, poi_data, station_demand, time_interval=None):
        """预测需求"""
        demands = []
        
        for _, grid in self.grids.iterrows():
            grid_id = grid['grid_id']
            
            # 基础需求 - 使用网格内站点的平均需求
            grid_stations = poi_data[poi_data['grid_id'] == grid_id]
            if len(grid_stations) > 0:
                station_ids = grid_stations['location_id'].unique()
                grid_base_demand = station_demand[station_demand.index.isin(station_ids)].mean()
                if pd.isna(grid_base_demand):
                    grid_base_demand = station_demand.mean()
            else:
                grid_base_demand = station_demand.mean() if len(station_demand) > 0 else 0
            
            # POI调整
            grid_poi = poi_data[poi_data['grid_id'] == grid_id]
            poi_adjustment = 0
            
            # 确保poi_weights是字典格式
            if self.poi_weights is not None and isinstance(self.poi_weights, dict):
                for category, weight in self.poi_weights.items():
                    count = len(grid_poi[grid_poi['category'] == category])
                    poi_adjustment += weight * count
            else:
                print("警告: POI权重格式不正确，跳过POI调整")
            
            # 时间调整
            time_factor = 1.0
            if time_interval is not None and self.time_factors is not None:
                time_factor = self.time_factors.get(time_interval, 1.0)
            
            final_demand = max(0, grid_base_demand * (1 + poi_adjustment) * time_factor)
            demands.append(final_demand)
        
        return np.array(demands)

class SiteSelectionModel:
    def __init__(self, grids, demands, budget, max_coverage_distance=500):
        self.grids = grids
        self.demands = demands
        self.budget = budget
        self.max_coverage_distance = max_coverage_distance
        self.covering_matrix = None
        
    def calculate_distance(self, grid1, grid2):
        """计算两个网格中心点之间的距离（米）"""
        lon1, lat1 = grid1['center_lon'], grid1['center_lat']
        lon2, lat2 = grid2['center_lon'], grid2['center_lat']
        
        # 简化的距离计算
        dx = (lon2 - lon1) * 111320 * np.cos(np.radians((lat1 + lat2) / 2))
        dy = (lat2 - lat1) * 111320
        return np.sqrt(dx**2 + dy**2)
    
    def build_covering_matrix(self):
        """构建覆盖矩阵"""
        n_grids = len(self.grids)
        self.covering_matrix = np.zeros((n_grids, n_grids), dtype=int)
        
        for i in range(n_grids):
            for j in range(n_grids):
                distance = self.calculate_distance(
                    self.grids.iloc[i], 
                    self.grids.iloc[j]
                )
                if distance <= self.max_coverage_distance:
                    self.covering_matrix[i, j] = 1
        
        print("覆盖矩阵构建完成")
        return self.covering_matrix
    
    def solve_mclp(self, construction_costs, lambda_weight=0.1):
        """求解最大覆盖位置问题"""
        n_grids = len(self.grids)
        
        # 创建优化问题
        prob = pulp.LpProblem("Site_Selection", pulp.LpMaximize)
        
        # 决策变量
        x = pulp.LpVariable.dicts("Build", range(n_grids), cat='Binary')
        cov = pulp.LpVariable.dicts("Cover", range(n_grids), cat='Binary')
        
        # 目标函数
        objective = pulp.lpSum([self.demands[i] * cov[i] for i in range(n_grids)]) - \
                   lambda_weight * pulp.lpSum([construction_costs[i] * x[i] for i in range(n_grids)])
        prob += objective
        
        # 预算约束
        prob += pulp.lpSum([construction_costs[i] * x[i] for i in range(n_grids)]) <= self.budget
        
        # 覆盖约束
        for j in range(n_grids):
            prob += pulp.lpSum([self.covering_matrix[i, j] * x[i] for i in range(n_grids)]) >= cov[j]
        
        # 求解
        print("开始求解站点选址问题...")
        prob.solve(pulp.PULP_CBC_CMD(msg=1))
        
        # 提取结果
        selected_sites = [i for i in range(n_grids) if pulp.value(x[i]) == 1]
        coverage = [pulp.value(cov[i]) for i in range(n_grids)]
        
        total_demand = np.sum(self.demands)
        covered_demand = np.sum(self.demands * coverage)
        
        print(f"选中的站点数量: {len(selected_sites)}")
        print(f"需求覆盖率: {covered_demand/total_demand:.2%}")
        
        return selected_sites, coverage

class PricingModel:
    def __init__(self, total_cost, base_demand, reference_price, price_elasticity):
        self.total_cost = total_cost
        self.base_demand = base_demand
        self.reference_price = reference_price
        self.price_elasticity = price_elasticity
        
    def demand_function(self, price, employer_subsidy):
        """需求函数"""
        net_price = price - employer_subsidy
        return self.base_demand * ((self.reference_price - net_price) / self.reference_price) ** self.price_elasticity
    
    def optimize_pricing(self, max_price, max_subsidy):
        """优化定价策略"""
        best_ridership = 0
        best_price = 0
        best_subsidy = 0
        
        # 网格搜索寻找最优解
        for price in np.linspace(0.1, max_price, 50):
            for subsidy in np.linspace(0, min(price, max_subsidy), 50):
                ridership = self.demand_function(price, subsidy)
                revenue = price * ridership + subsidy * ridership
                
                # 检查收入是否覆盖成本
                if revenue >= self.total_cost:
                    if ridership > best_ridership:
                        best_ridership = ridership
                        best_price = price
                        best_subsidy = subsidy
        
        return best_price, best_subsidy, best_ridership

def main():
    print("=== 爱丁堡自行车共享系统建模 ===")
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    loader = EdinburghBikeDataLoader()
    
    stations = loader.load_station_data()
    pois = loader.load_poi_data()
    cycle_hire = loader.load_cycle_hire_data()
    counts = loader.load_counts_data()
    
    # 2. 数据预处理和分析
    print("\n2. 数据预处理和分析...")
    
    if stations is not None:
        print(f"站点数据: {len(stations)} 个站点")
        print(f"纬度范围: {stations['lat'].min():.4f} - {stations['lat'].max():.4f}")
        print(f"经度范围: {stations['lon'].min():.4f} - {stations['lon'].max():.4f}")
    
    if pois is not None:
        print(f"POI数据: {len(pois)} 个POI，类别: {pois['category'].unique()}")
    
    if counts is not None:
        print(f"计数数据: {len(counts)} 条记录，总行程: {counts['trip_count'].sum()}")
    
    # 3. 创建网格系统
    print("\n3. 创建网格系统...")
    if stations is not None:
        # 计算边界（稍微扩展以确保覆盖所有点）
        buffer = 0.01
        bounds = (
            stations['lon'].min() - buffer,
            stations['lon'].max() + buffer,
            stations['lat'].min() - buffer,
            stations['lat'].max() + buffer
        )
        
        model = BikeSharingModel(grid_size=500)
        grids = model.create_grid_system(bounds)
        
        # 4. 分配位置到网格 - 修复版本
        print("\n4. 分配位置到网格...")
        if stations is not None:
            station_assignments = model.assign_locations_to_grids(stations, 'lon', 'lat')
            print(f"分配了 {len(station_assignments)} 个站点到网格")
        
        if pois is not None:
            # 保留类别信息
            poi_assignments = model.assign_locations_to_grids(pois, 'lon', 'lat', keep_columns=['category'])
            print(f"分配了 {len(poi_assignments)} 个POI到网格")
            
            # 检查是否成功保留了类别信息
            if 'category' in poi_assignments.columns:
                print(f"成功保留POI类别信息，共 {len(poi_assignments['category'].unique())} 个类别")
            else:
                print("警告：POI分配结果中缺少类别信息")
        
        # 5. 需求估计
        print("\n5. 需求估计...")
        demand_model = DemandModel(grids)
        
        # 计算基础需求
        if counts is not None:
            base_demand = demand_model.calculate_base_demand(counts)
            print(f"计算了 {len(base_demand)} 个站点的基准需求")
        
        # 计算POI权重 - 使用修复版本
        if pois is not None and counts is not None:
            poi_weights = demand_model.calculate_poi_weights(poi_assignments, counts)
            print("POI权重:", poi_weights)
        
        # 计算时间因子
        if counts is not None:
            time_factors = demand_model.calculate_time_factors(counts)
            print("时间因子计算完成")
        
        # 预测总需求
        if counts is not None and pois is not None:
            total_demand = demand_model.predict_demand(poi_assignments, base_demand)
            print(f"总需求预测完成: 平均 {np.mean(total_demand):.2f}, 总计 {np.sum(total_demand):.0f}")
        
        # 6. 站点选址
        print("\n6. 站点选址...")
        budget = 500000  # 50万预算
        construction_costs = np.full(len(grids), 5000)  # 每个站点5000成本
        
        site_model = SiteSelectionModel(grids, total_demand, budget)
        covering_matrix = site_model.build_covering_matrix()
        selected_sites, coverage = site_model.solve_mclp(construction_costs)
        
        # 7. 定价模型
        print("\n7. 定价优化...")
        total_operational_cost = len(selected_sites) * 5000 + 100000  # 简化成本计算
        
        pricing_model = PricingModel(
            total_cost=total_operational_cost,
            base_demand=np.sum(total_demand),
            reference_price=2.0,
            price_elasticity=-1.5
        )
        
        optimal_price, optimal_subsidy, optimal_ridership = pricing_model.optimize_pricing(
            max_price=5.0, max_subsidy=2.0
        )
        
        print(f"\n=== 最终结果 ===")
        print(f"最优定价策略:")
        print(f"  用户价格: £{optimal_price:.2f}")
        print(f"  雇主补贴: £{optimal_subsidy:.2f}")
        print(f"  预计骑行量: {optimal_ridership:.0f}")
        print(f"  总收入: £{(optimal_price + optimal_subsidy) * optimal_ridership:.2f}")
        print(f"  总成本: £{total_operational_cost:.2f}")
        
        # 8. 可视化结果
        print("\n8. 生成可视化...")
        visualize_results(grids, total_demand, selected_sites, coverage)
        
    else:
        print("无法创建网格系统：缺少站点数据")

def visualize_results(grids, demands, selected_sites, coverage):
    """可视化结果"""
    try:
        plt.figure(figsize=(15, 10))
        
        # 绘制需求热力图
        plt.subplot(2, 2, 1)
        plt.scatter(grids['center_lon'], grids['center_lat'], c=demands, cmap='YlOrRd', s=50)
        plt.colorbar(label='需求')
        plt.title('网格需求分布')
        plt.xlabel('经度')
        plt.ylabel('纬度')
        
        # 绘制选中的站点
        plt.subplot(2, 2, 2)
        plt.scatter(grids['center_lon'], grids['center_lat'], c='lightgray', s=10, alpha=0.5)
        plt.scatter(grids.iloc[selected_sites]['center_lon'], 
                   grids.iloc[selected_sites]['center_lat'], 
                   c='red', s=50, label='选中站点')
        plt.title('选中的站点位置')
        plt.xlabel('经度')
        plt.ylabel('纬度')
        plt.legend()
        
        # 绘制覆盖情况
        plt.subplot(2, 2, 3)
        covered_grids = grids.iloc[[i for i, cov in enumerate(coverage) if cov == 1]]
        uncovered_grids = grids.iloc[[i for i, cov in enumerate(coverage) if cov == 0]]
        
        plt.scatter(uncovered_grids['center_lon'], uncovered_grids['center_lat'], 
                   c='lightgray', s=10, alpha=0.5, label='未覆盖')
        plt.scatter(covered_grids['center_lon'], covered_grids['center_lat'], 
                   c='green', s=30, alpha=0.7, label='已覆盖')
        plt.scatter(grids.iloc[selected_sites]['center_lon'], 
                   grids.iloc[selected_sites]['center_lat'], 
                   c='red', s=50, marker='x', label='站点')
        plt.title('需求覆盖情况')
        plt.xlabel('经度')
        plt.ylabel('纬度')
        plt.legend()
        
        # 绘制需求分布直方图
        plt.subplot(2, 2, 4)
        plt.hist(demands, bins=30, alpha=0.7, edgecolor='black')
        plt.title('需求分布直方图')
        plt.xlabel('需求')
        plt.ylabel('网格数量')
        
        plt.tight_layout()
        plt.savefig('modeling_results.png', dpi=300, bbox_inches='tight')
        print("结果已保存到 modeling_results.png")
        
    except Exception as e:
        print(f"可视化时出错: {e}")

if __name__ == "__main__":
    main()