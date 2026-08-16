"""
Spatiotemporal Wildfire Spread Prediction & Dynamic Graph Routing Engine


Description:
------------
An end-to-end geospatial AI/ML engine combining predictive fire behavior modeling
with dynamic graph pathfinding for disaster evacuation.
Performs:
  1. Spatial Road Network Topology & Intersection Graph Construction.
  2. Environmental Feature Ingestion (Slope, Wind Velocity, Fuel Moisture, NDVI).
  3. Supervised Machine Learning Fire Spread Predictor (XGBoost Classifier).
  4. Spatiotemporal Fire Propagation Forecasting & DBSCAN Plume Clustering.
  5. Predictive Hazard Buffer Envelope & Spatial Risk Surface Generation.
  6. Dynamic Risk-Penalized Shortest Path Optimization via NetworkX/Dijkstra.
"""

from typing import Dict, List, Tuple, Any
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.cluster import DBSCAN
import xgboost as xgb
import networkx as nx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class WildfireAIEvacuationEngine:
    """
    Geospatial AI/ML pipeline for predictive wildfire spread modeling
    and dynamic hazard-aware evacuation routing.
    """

    def __init__(
        self,
        num_grid_cells: int = 1500,
        num_nodes: int = 80,
        random_state: int = 42
    ):
        """
        Initialize the AI/ML wildfire evacuation engine.

        :param num_grid_cells: Spatial cells for training fire spread ML model.
        :param num_nodes: Number of intersection nodes in the transit network.
        :param random_state: Seed for reproducible spatial synthesis.
        """
        self.num_grid_cells = num_grid_cells
        self.num_nodes = num_nodes
        self.random_state = random_state

        self.df_terrain: pd.DataFrame = pd.DataFrame()
        self.spread_model: xgb.XGBClassifier = None
        self.graph: nx.DiGraph = nx.DiGraph()
        self.nodes_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame()
        self.predicted_hazard_polygon: Polygon = None

    # =========================================================================
    # MODULE 1: Environmental Feature Ingestion & ML Spread Predictor
    # =========================================================================
    def build_environmental_feature_store(self) -> pd.DataFrame:
        """
        Simulates environmental rasters: Slope, Aspect, Wind Speed, Fuel Moisture, NDVI,
        and current thermal FRP metrics.
        """
        logger.info("Synthesizing environmental feature store (%d spatial grid cells)...", self.num_grid_cells)
        np.random.seed(self.random_state)

        base_lat, base_lon = 34.05, -118.35
        lats = base_lat + np.random.uniform(-0.05, 0.05, self.num_grid_cells)
        lons = base_lon + np.random.uniform(-0.05, 0.05, self.num_grid_cells)

        # Environmental covariates
        slope_deg = np.random.uniform(0.0, 35.0, self.num_grid_cells)
        wind_speed_kmh = np.random.uniform(5.0, 65.0, self.num_grid_cells)
        wind_alignment = np.random.uniform(-1.0, 1.0, self.num_grid_cells)  # Cosine angle with wind direction
        fuel_moisture_pct = np.random.uniform(4.0, 25.0, self.num_grid_cells)
        ndvi = np.random.uniform(0.1, 0.85, self.num_grid_cells)
        current_frp_mw = np.clip(np.random.exponential(scale=15.0, size=self.num_grid_cells) - 8.0, 0.0, 350.0)

        # Non-linear Rothermel-inspired ground truth fire spread logic:
        # High spread probability if steep slope, high wind aligned with spread, dry fuel, dense vegetation
        spread_logit = (
            0.08 * slope_deg +
            0.06 * wind_speed_kmh * np.maximum(0, wind_alignment) +
            0.02 * current_frp_mw -
            0.25 * fuel_moisture_pct +
            2.5 * ndvi -
            2.0
        )
        spread_prob = 1.0 / (1.0 + np.exp(-spread_logit))
        will_spread_next_hour = (np.random.rand(self.num_grid_cells) < spread_prob).astype(int)

        self.df_terrain = pd.DataFrame({
            "Cell_ID": [f"CELL_{i:04d}" for i in range(self.num_grid_cells)],
            "Latitude": lats,
            "Longitude": lons,
            "Slope_deg": np.round(slope_deg, 2),
            "WindSpeed_kmh": np.round(wind_speed_kmh, 2),
            "Wind_Alignment": np.round(wind_alignment, 3),
            "FuelMoisture_Pct": np.round(fuel_moisture_pct, 2),
            "NDVI": np.round(ndvi, 3),
            "Current_FRP_MW": np.round(current_frp_mw, 2),
            "Will_Spread": will_spread_next_hour
        })

        logger.info("Feature store generated: Positive fire spread rate = %.2f%%",
                    self.df_terrain["Will_Spread"].mean() * 100)
        return self.df_terrain

    def train_wildfire_spread_model(self) -> Dict[str, float]:
        """
        Trains an XGBoost classifier predicting wildfire propagation probability at t + 1 hour.
        """
        logger.info("Training predictive XGBoost Wildfire Spread Classifier...")
        feature_cols = [
            "Slope_deg", "WindSpeed_kmh", "Wind_Alignment",
            "FuelMoisture_Pct", "NDVI", "Current_FRP_MW"
        ]
        X = self.df_terrain[feature_cols]
        y = self.df_terrain["Will_Spread"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=self.random_state
        )

        self.spread_model = xgb.XGBClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=self.random_state
        )
        self.spread_model.fit(X_train, y_train)

        preds = self.spread_model.predict(X_test)
        probs = self.spread_model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, probs)
        logger.info("Model Training Complete -> Test ROC-AUC: %.4f", auc)
        print("\n=== Predictive Wildfire Spread Classification Report ===")
        print(classification_report(y_test, preds, digits=4))

        # Add predicted spread probabilities back to spatial DataFrame
        self.df_terrain["Predicted_Spread_Prob"] = self.spread_model.predict_proba(X)[:, 1]
        return {"ROC_AUC": auc}

    # =========================================================================
    # MODULE 2: Spatial Spatiotemporal Hazard Envelope Construction
    # =========================================================================
    def construct_predictive_hazard_polygon(self, prob_threshold: float = 0.60) -> Polygon:
        """
        Filters cells exceeding high spread probability, clusters them via DBSCAN,
        and constructs a spatial buffer representing the predicted fire front at t+1 hr.
        """
        logger.info("Extracting high-risk propagation zones (Spread Probability >= %.2f)...", prob_threshold)
        high_risk_cells = self.df_terrain[self.df_terrain["Predicted_Spread_Prob"] >= prob_threshold].copy()

        if len(high_risk_cells) == 0:
            logger.warning("No cells exceeded probability threshold; defaulting to top 10% highest risk.")
            thresh = self.df_terrain["Predicted_Spread_Prob"].quantile(0.90)
            high_risk_cells = self.df_terrain[self.df_terrain["Predicted_Spread_Prob"] >= thresh].copy()

        # Spatial clustering via DBSCAN with Haversine metric
        kms_per_radian = 6371.0088
        eps_rad = 1.0 / kms_per_radian  # 1.0 km neighborhood
        coords_rad = np.radians(high_risk_cells[["Latitude", "Longitude"]])

        db = DBSCAN(eps=eps_rad, min_samples=3, metric="haversine").fit(coords_rad)
        high_risk_cells["Cluster"] = db.labels_

        # Exclude unclustered noise points (-1) if valid clusters exist
        valid_clusters = high_risk_cells[high_risk_cells["Cluster"] != -1]
        if len(valid_clusters) == 0:
            valid_clusters = high_risk_cells

        risk_gdf = gpd.GeoDataFrame(
            valid_clusters,
            geometry=[Point(xy) for xy in zip(valid_clusters["Longitude"], valid_clusters["Latitude"])],
            crs="EPSG:4326"
        ).to_crs(epsg=3857)

        # Buffer predicted front by 500m to generate safety envelope
        self.predicted_hazard_polygon = risk_gdf.buffer(500.0).unary_union
        logger.info("Constructed Predictive Hazard Buffer Polygon across %d high-risk cells.", len(valid_clusters))
        return self.predicted_hazard_polygon

    # =========================================================================
    # MODULE 3: Network Topology & Dynamic ML-Penalized Routing
    # =========================================================================
    def construct_road_network(self) -> nx.DiGraph:
        """Constructs a directed road network graph with spatial node coordinates."""
        logger.info("Building topological road network with %d nodes...", self.num_nodes)
        np.random.seed(self.random_state)

        base_lat, base_lon = 34.05, -118.35
        node_ids = list(range(self.num_nodes))
        lats = base_lat + np.random.uniform(-0.045, 0.045, self.num_nodes)
        lons = base_lon + np.random.uniform(-0.045, 0.045, self.num_nodes)

        nodes_dict = {}
        for i, nid in enumerate(node_ids):
            nodes_dict[nid] = {"x": lons[i], "y": lats[i]}
            self.graph.add_node(nid, x=lons[i], y=lats[i])

        self.nodes_gdf = gpd.GeoDataFrame(
            {"node_id": node_ids},
            geometry=[Point(lons[i], lats[i]) for i in range(self.num_nodes)],
            crs="EPSG:4326"
        )

        for u in range(self.num_nodes):
            distances = [
                (v, np.hypot(nodes_dict[u]["x"] - nodes_dict[v]["x"], nodes_dict[u]["y"] - nodes_dict[v]["y"]))
                for v in range(self.num_nodes) if u != v
            ]
            distances.sort(key=lambda item: item[1])
            for v, dist in distances[:3]:
                length_m = dist * 111000.0  # Approx meters
                self.graph.add_edge(u, v, length=length_m, dynamic_cost=length_m)

        return self.graph

    def compute_ai_evacuation_routing(
        self,
        origin_node: int = 0,
        destination_node: int = 79
    ) -> Dict[str, Any]:
        """
        Evaluates dynamic edge impedances against ML-predicted wildfire hazard polygons
        and solves for the optimal safe route via Dijkstra.
        """
        logger.info("Re-weighting edge costs based on ML predicted hazard envelope...")
        nodes_proj = self.nodes_gdf.to_crs(epsg=3857).set_index("node_id")

        # 1. Baseline Route (Unaware of fire spread)
        base_route = nx.shortest_path(self.graph, source=origin_node, target=destination_node, weight="length")
        base_dist = nx.shortest_path_length(self.graph, source=origin_node, target=destination_node, weight="length")

        # 2. Re-weight edges intersecting the ML predicted hazard envelope
        penalized_count = 0
        for u, v, data in self.graph.edges(data=True):
            p1 = nodes_proj.loc[u].geometry
            p2 = nodes_proj.loc[v].geometry
            segment = LineString([p1, p2])

            if segment.intersects(self.predicted_hazard_polygon):
                data["dynamic_cost"] = data["length"] * 120.0  # Heavy dynamic penalty
                penalized_count += 1
            else:
                data["dynamic_cost"] = data["length"]

        logger.info("Imposed ML-risk penalties on %d road segments.", penalized_count)

        # 3. AI Predictive Safe Route
        safe_route = nx.shortest_path(self.graph, source=origin_node, target=destination_node, weight="dynamic_cost")
        safe_dist = nx.shortest_path_length(self.graph, source=origin_node, target=destination_node, weight="length")

        base_breach = any(
            LineString([nodes_proj.loc[u].geometry, nodes_proj.loc[v].geometry]).intersects(self.predicted_hazard_polygon)
            for u, v in zip(base_route[:-1], base_route[1:])
        )
        safe_breach = any(
            LineString([nodes_proj.loc[u].geometry, nodes_proj.loc[v].geometry]).intersects(self.predicted_hazard_polygon)
            for u, v in zip(safe_route[:-1], safe_route[1:])
        )

        return {
            "origin_node": origin_node,
            "destination_node": destination_node,
            "base_route": base_route,
            "base_distance_m": base_dist,
            "base_breach": base_breach,
            "safe_route": safe_route,
            "safe_distance_m": safe_dist,
            "safe_breach": safe_breach
        }


def main():
    """Execution entry point."""
    print("=" * 85)
    print("  SPATIOTEMPORAL WILDFIRE SPREAD ML & PREDICTIVE EVACUATION ENGINE")
    print("=" * 85)

    engine = WildfireAIEvacuationEngine(num_grid_cells=1500, num_nodes=80, random_state=42)

    # 1. Ingest Terrain & Environmental Covariates
    engine.build_environmental_feature_store()

    # 2. Train XGBoost Wildfire Spread Predictor
    engine.train_wildfire_spread_model()

    # 3. Construct Predictive Fire Spread Hazard Buffer
    engine.construct_predictive_hazard_polygon(prob_threshold=0.60)

    # 4. Road Network Topology
    engine.construct_road_network()

    # 5. Execute Dynamic AI Evacuation Routing
    result = engine.compute_ai_evacuation_routing(origin_node=0, destination_node=79)

    print("\n" + "=" * 85)
    print("              PREDICTIVE AI EVACUATION ROUTING AUDIT")
    print("=" * 85)
    print(f"Origin Node                 : {result['origin_node']}")
    print(f"Destination Node            : {result['destination_node']}")
    print("-" * 85)
    print(f"Standard Shortest Path      : {result['base_route']}")
    print(f"Standard Path Distance      : {result['base_distance_m']:.2f} meters")
    print(f"Predicted Fire Front Breach : {'CRITICAL HAZARD DETECTED' if result['base_breach'] else 'SAFE'}")
    print("-" * 85)
    print(f"AI Hazard-Aware Safe Route  : {result['safe_route']}")
    print(f"AI Path Distance            : {result['safe_distance_m']:.2f} meters")
    print(f"Predicted Fire Front Breach : {'CRITICAL HAZARD' if result['safe_breach'] else 'ZERO EXPOSURE (SAFE EVACUATION)'}")
    print("=" * 85 + "\n")
    print("[SUCCESS] Geospatial AI wildfire routing pipeline completed with 0 errors.")


if __name__ == "__main__":
    main()
