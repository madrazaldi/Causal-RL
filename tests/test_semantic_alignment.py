from __future__ import annotations

import json

import pytest

from causal_rl.config import (
    CAUSAL_BACKDOOR_COLUMNS,
    DATA_DICTIONARY_PATH,
    LATENT_COLUMNS,
    NON_CAUSAL_EXTRA_COLUMNS,
    POST_ACTION_COLUMNS,
    RAW_DATA_PATH,
    SEMANTIC_REFERENCE,
)
from causal_rl.dataset import build_decision_log, load_raw_data


@pytest.fixture(scope="module")
def raw_df():
    return load_raw_data(RAW_DATA_PATH)


@pytest.fixture(scope="module")
def bundle(raw_df):
    return build_decision_log(raw_df)


@pytest.fixture(scope="module")
def data_dictionary():
    with DATA_DICTIONARY_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def test_raw_csv_columns_match_dictionary_schema(raw_df, data_dictionary) -> None:
    dictionary_columns = list(data_dictionary["variables"].keys())
    raw_columns = [column for column in raw_df.columns if column != "row_id"]

    assert raw_columns == dictionary_columns


def test_real_categorical_levels_match_dataset_and_dictionary(raw_df, data_dictionary) -> None:
    assert data_dictionary["variables"]["vehicle_type"]["description"] == 'Type of vehicle "common" or "cooling" truck'
    assert sorted(raw_df["vehicle_type"].dropna().unique().tolist()) == ["common", "cooling"]
    assert sorted(raw_df["traffic_state"].dropna().unique().tolist()) == ["high", "low", "medium"]
    assert sorted(raw_df["commodity_type"].dropna().unique().tolist()) == ["cold_chain", "general"]


def test_state_roles_match_semantic_reference(bundle) -> None:
    metadata = bundle.metadata

    assert metadata["semantic_reference"] == SEMANTIC_REFERENCE
    assert metadata["semantic_reference"]["action_semantics"].endswith("may change across the day.")
    assert set(CAUSAL_BACKDOOR_COLUMNS).isdisjoint(POST_ACTION_COLUMNS + LATENT_COLUMNS + NON_CAUSAL_EXTRA_COLUMNS)
    assert set(NON_CAUSAL_EXTRA_COLUMNS).issubset(metadata["non_causal_state_columns"])
    assert set(NON_CAUSAL_EXTRA_COLUMNS).isdisjoint(metadata["causal_state_columns"])


def test_metadata_stays_in_sync_with_config(bundle) -> None:
    metadata = bundle.metadata

    assert metadata["post_action_columns"] == POST_ACTION_COLUMNS
    assert metadata["latent_columns"] == LATENT_COLUMNS
    assert metadata["deployable_exclusions"] == POST_ACTION_COLUMNS + LATENT_COLUMNS
    assert metadata["causal_state_columns"] == CAUSAL_BACKDOOR_COLUMNS
    assert metadata["sequence_ordering"]["sort_columns"] == ["date", "vehicle_id", "hour", "row_id"]
    assert metadata["sequence_ordering"]["duplicate_date_vehicle_hour_groups"] > 0
    assert metadata["sequence_ordering"]["duplicate_date_vehicle_hour_group_share"] > 0.15
    assert metadata["sequence_ordering"]["max_rows_per_date_vehicle_hour"] >= 2


def test_raw_schema_has_no_finer_grained_order_column(raw_df, bundle) -> None:
    candidate_columns = {
        "timestamp",
        "datetime",
        "decision_time",
        "decision_ts",
        "departure_time",
        "arrival_time",
        "minute",
        "second",
    }

    assert candidate_columns.isdisjoint(raw_df.columns)

    ordered = bundle.df.sort_values(["date", "vehicle_id", "hour", "row_id"])
    monotonic_by_group = ordered.groupby(["date", "vehicle_id", "hour"], sort=False)["row_id"].apply(
        lambda series: series.is_monotonic_increasing
    )
    assert monotonic_by_group.all()


def test_trajectory_grouping_is_date_vehicle_not_zone(bundle) -> None:
    trajectory_per_date_vehicle = bundle.df.groupby(["date", "vehicle_id"], sort=False)["trajectory_id"].nunique()
    trajectory_per_date_vehicle_zone = bundle.df.groupby(["date", "vehicle_id", "zone"], sort=False)["trajectory_id"].nunique()

    assert (trajectory_per_date_vehicle == 1).all()
    assert (trajectory_per_date_vehicle_zone == 1).all()
    assert bundle.metadata["semantic_reference"]["trajectory_definition"] == "A daily trajectory is defined by the (date, vehicle_id) pair."
