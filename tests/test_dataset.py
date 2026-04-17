from __future__ import annotations

import pandas as pd

from causal_rl.dataset import build_decision_log


def make_raw_frame() -> pd.DataFrame:
    rows = []
    base_rows = [
        ("2025-01-01", 0, 2, 9, 1, 10, "common", 2.0, 0.95, -0.1, 0.2, 0.1, "general", 2.5, 0.2, 8.0, 40, 12, -0.5, 0.1, 0, 0.0, 15.0, 10.0, 0, 0, 0.4, "light", 0, 1, 3.0, 0, 5.0, 30.0, 11.0, 0, 1.2, 2.5, -1.0, 0, 0, 0.0, 1),
        ("2025-01-01", 0, 2, 11, 1, 10, "common", 2.0, 0.95, -0.1, 0.2, 0.1, "general", 2.8, 0.3, 10.0, 40, 12, -0.5, 0.1, 0, 0.0, 15.5, 10.0, 0, 0, 0.7, "medium", 0, 0, 4.0, 0, 6.0, 28.0, 13.0, 0, 1.4, 2.8, -0.8, 1, 0, 4.0, 0),
        ("2025-01-02", 1, 3, 9, 2, 11, "cooling", 3.0, 0.93, 0.0, 0.3, 0.2, "cold", 3.0, 0.5, 12.0, 35, 14, -0.3, 0.2, 1, 5.0, 9.0, 8.0, 1, 0, 1.2, "heavy", 1, 0, 8.0, 1, 8.0, 22.0, 20.0, 1, 1.8, 3.8, -0.2, 0, 0, 6.0, 0),
        ("2025-01-02", 1, 3, 14, 2, 11, "cooling", 3.0, 0.93, 0.0, 0.3, 0.2, "cold", 3.2, 0.6, 14.0, 35, 14, -0.3, 0.2, 1, 4.0, 10.0, 8.5, 0, 1, 1.0, "heavy", 1, 1, 6.0, 1, 7.0, 24.0, 18.0, 1, 1.6, 3.5, -0.1, 0, 1, 8.0, 0),
    ]
    columns = [
        "date",
        "day_idx",
        "dow",
        "hour",
        "zone",
        "vehicle_id",
        "vehicle_type",
        "vehicle_age_years",
        "vehicle_efficiency_index",
        "maintenance_latent",
        "driver_skill_latent",
        "driver_risk_propensity_latent",
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
        "eco_mode",
        "dispatch_delay_min",
        "compatibility_violation",
        "distance_km",
        "avg_speed_kmph",
        "travel_time_min",
        "harsh_events",
        "fuel_liters",
        "co2_kg",
        "risk_score",
        "near_miss",
        "crash",
        "lateness_min",
        "on_time",
    ]
    for row in base_rows:
        rows.append(dict(zip(columns, row)))
    return pd.DataFrame(rows)


def test_build_decision_log_has_unique_trajectory_assignment() -> None:
    bundle = build_decision_log(make_raw_frame())
    df = bundle.df

    assert len(df) == 4
    assert df["trajectory_id"].nunique() == 2
    assert (df.groupby("trajectory_id")["t"].min() == 0).all()
    assert (df.groupby("trajectory_id")["done"].sum() == 1).all()


def test_build_decision_log_excludes_post_action_state_leakage() -> None:
    bundle = build_decision_log(make_raw_frame())
    state_columns = bundle.metadata["state_columns"]

    for leaked in ["co2_kg", "crash", "near_miss", "lateness_min", "on_time", "harsh_events"]:
        assert leaked not in state_columns
        assert f"state_{leaked}" not in bundle.df.columns


def test_temporal_split_is_date_based() -> None:
    bundle = build_decision_log(make_raw_frame())
    df = bundle.df

    split_by_date = df.groupby(df["date"].dt.strftime("%Y-%m-%d"))["split"].nunique()
    assert (split_by_date == 1).all()
