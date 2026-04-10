from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT / "causalog_synthetic_urban_logistics.csv"

ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

DECISION_LOG_PATH = ARTIFACTS_DIR / "decision_log.csv"
METADATA_PATH = ARTIFACTS_DIR / "dataset_metadata.json"
POLICY_SUMMARY_PATH = MODELS_DIR / "policy_summary.json"
POLICY_ACTIONS_PATH = RESULTS_DIR / "policy_actions.csv"
METRICS_PATH = RESULTS_DIR / "metrics.csv"
ROBUSTNESS_PATH = RESULTS_DIR / "robustness.csv"
REWARD_SENSITIVITY_PATH = RESULTS_DIR / "reward_sensitivity.csv"
MAIN_RESULTS_TABLE_PATH = RESULTS_DIR / "main_results_table.csv"

SEED = 42
GAMMA = 0.95
FQI_ITERATIONS = 5
MIN_PROPENSITY = 0.10
Q_GAP_THRESHOLD = 0.25

SPLIT_RATIOS = (0.70, 0.15, 0.15)

ACTION_COLUMN = "eco_mode"
DATE_COLUMN = "date"
TRAJECTORY_KEY_COLUMNS = [DATE_COLUMN, "vehicle_id"]
SORT_COLUMNS = [DATE_COLUMN, "vehicle_id", "hour", "row_id"]

POST_ACTION_COLUMNS = [
    "avg_speed_kmph",
    "travel_time_min",
    "fuel_liters",
    "co2_kg",
    "near_miss",
    "crash",
    "lateness_min",
    "on_time",
]

LATENT_COLUMNS = [
    "maintenance_latent",
    "driver_skill_latent",
    "driver_risk_propensity_latent",
]

BASE_STATE_COLUMNS = [
    "day_idx",
    "dow",
    "hour",
    "zone",
    "vehicle_id",
    "vehicle_type",
    "vehicle_age_years",
    "vehicle_efficiency_index",
    "commodity_type",
    "demand_size",
    "time_window_tightness",
    "service_time_min",
    "speed_limit_kmph",
    "intersection_density",
    "road_grade_index",
    "road_risk_index",
    "rain",
    "rain_intensity",
    "temperature_c",
    "visibility_km",
    "event_indicator",
    "roadworks_indicator",
    "traffic_index",
    "traffic_state",
    "route_risky",
    "dispatch_delay_min",
    "compatibility_violation",
    "distance_km",
    "risk_score",
]

SEQUENTIAL_STATE_COLUMNS = [
    "step_idx",
    "remaining_steps",
    "rolling_mean_traffic",
    "rolling_cumulative_lateness",
    "rolling_incident_count",
    "prior_reward_primary",
    "prior_eco_mode",
]

DEPLOYABLE_STATE_COLUMNS = BASE_STATE_COLUMNS + SEQUENTIAL_STATE_COLUMNS

CAUSAL_BACKDOOR_COLUMNS = [
    "day_idx",
    "dow",
    "hour",
    "zone",
    "vehicle_id",
    "vehicle_type",
    "vehicle_age_years",
    "vehicle_efficiency_index",
    "commodity_type",
    "demand_size",
    "time_window_tightness",
    "service_time_min",
    "speed_limit_kmph",
    "intersection_density",
    "road_grade_index",
    "road_risk_index",
    "rain",
    "rain_intensity",
    "temperature_c",
    "visibility_km",
    "event_indicator",
    "roadworks_indicator",
    "traffic_index",
    "traffic_state",
    "route_risky",
    "dispatch_delay_min",
    "step_idx",
    "remaining_steps",
    "rolling_mean_traffic",
    "rolling_cumulative_lateness",
    "rolling_incident_count",
    "prior_reward_primary",
    "prior_eco_mode",
]

NON_CAUSAL_EXTRA_COLUMNS = ["compatibility_violation", "distance_km", "risk_score"]

REWARD_COLUMNS = {
    "primary": "reward_primary",
    "service_heavy": "reward_service_heavy",
    "sustainability_heavy": "reward_sustainability_heavy",
}

REWARD_SPECS = {
    "primary": {"lateness_min": 1.0, "crash": 10.0, "near_miss": 2.0, "co2_kg": 0.2},
    "service_heavy": {"lateness_min": 1.5, "crash": 12.0, "near_miss": 3.0, "co2_kg": 0.1},
    "sustainability_heavy": {"lateness_min": 0.8, "crash": 10.0, "near_miss": 2.0, "co2_kg": 0.4},
}

OUTCOME_MODEL_TARGETS = [
    "lateness_min",
    "co2_kg",
    "crash",
    "near_miss",
    "on_time",
]

ROBUSTNESS_SEGMENTS = {
    "high_traffic": "traffic_index >= traffic_index.quantile(0.75)",
    "rain_or_event": "(rain == 1) or (event_indicator == 1)",
    "tight_window": "time_window_tightness >= time_window_tightness.quantile(0.75)",
    "late_day": "hour >= 17",
}
