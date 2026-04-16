from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    ACTION_COLUMN,
    ABLATION_STATE_REGISTRY,
    DECISION_LOG_PATH,
    FQI_ITERATIONS,
    GAMMA,
    LEARNED_POLICY_REGISTRY,
    MAIN_RESULTS_TABLE_PATH,
    METADATA_PATH,
    MIN_PROPENSITY,
    MODELS_DIR,
    OUTCOME_MODEL_TARGETS,
    POLICY_SUMMARY_PATH,
    Q_GAP_THRESHOLD,
    REWARD_COLUMNS,
    REWARD_SENSITIVITY_PATH,
    SEED,
)


@dataclass
class PolicyArtifacts:
    behavior_model: Pipeline
    heuristic_model: Pipeline
    outcome_targets: dict[str, Pipeline]
    learned_policies: dict[str, "FittedQPolicy"]
    metadata: dict


def load_training_frame(path: Path = DECISION_LOG_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def load_metadata(path: Path = METADATA_PATH) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_preprocessor(df: pd.DataFrame, columns: list[str]) -> ColumnTransformer:
    categorical = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
    numeric = [c for c in columns if c not in categorical]
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ]
    )


def make_behavior_model(df: pd.DataFrame, columns: list[str]) -> Pipeline:
    preprocessor = build_preprocessor(df, columns)
    base = LogisticRegression(max_iter=1000, random_state=SEED)
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    return Pipeline([("preprocessor", preprocessor), ("model", clf)])


def make_regressor_pipeline(df: pd.DataFrame, columns: list[str], random_state: int = SEED) -> Pipeline:
    preprocessor = build_preprocessor(df, columns)
    regressor = HistGradientBoostingRegressor(
        max_depth=6,
        max_iter=150,
        learning_rate=0.05,
        random_state=random_state,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", regressor)])


class FittedQPolicy:
    def __init__(
        self,
        state_columns: list[str],
        gamma: float = GAMMA,
        iterations: int = FQI_ITERATIONS,
        name: str = "fqi",
    ) -> None:
        self.state_columns = state_columns
        self.gamma = gamma
        self.iterations = iterations
        self.name = name
        self.preprocessor: ColumnTransformer | None = None
        self.models: dict[int, object] = {}
        self.training_history: list[float] = []

    def fit(self, df: pd.DataFrame, reward_column: str = REWARD_COLUMNS["primary"]) -> "FittedQPolicy":
        self.preprocessor = build_preprocessor(df, self.state_columns)
        X_state = self.preprocessor.fit_transform(df[self.state_columns])
        next_state_columns = [f"next_state_{col}" for col in self.state_columns]
        X_next = self.preprocessor.transform(
            df[next_state_columns].rename(columns={f"next_state_{c}": c for c in self.state_columns})
        )
        rewards = df[reward_column].to_numpy(dtype=float)
        actions = df["action"].to_numpy(dtype=int)
        done = df["done"].to_numpy(dtype=int)
        q_next = np.zeros((len(df), 2), dtype=float)

        for iteration in range(self.iterations):
            targets = rewards + self.gamma * (1.0 - done) * q_next.max(axis=1)
            for action in (0, 1):
                mask = actions == action
                model = HistGradientBoostingRegressor(
                    max_depth=6,
                    max_iter=120,
                    learning_rate=0.05,
                    random_state=SEED + iteration * 11 + action,
                )
                model.fit(X_state[mask], targets[mask])
                self.models[action] = model
            q_next = np.column_stack([self.models[a].predict(X_next) for a in (0, 1)])
            self.training_history.append(float(np.mean(targets)))
        return self

    def q_values(self, df: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is None:
            raise RuntimeError("Policy must be fit before scoring.")
        X = self.preprocessor.transform(df[self.state_columns])
        return np.column_stack([self.models[a].predict(X) for a in (0, 1)])

    def greedy_action(self, df: pd.DataFrame) -> np.ndarray:
        q_values = self.q_values(df)
        return q_values.argmax(axis=1).astype(int)

    def policy_action(
        self,
        df: pd.DataFrame,
        behavior_model: Pipeline,
        behavior_state_columns: list[str],
        fallback_to_logged: bool = True,
        min_propensity: float = MIN_PROPENSITY,
        q_gap_threshold: float = Q_GAP_THRESHOLD,
    ) -> pd.DataFrame:
        q_values = self.q_values(df)
        greedy = q_values.argmax(axis=1).astype(int)
        q_gap = np.abs(q_values[:, 1] - q_values[:, 0])
        prop = behavior_model.predict_proba(df[behavior_state_columns])
        chosen_propensity = prop[np.arange(len(df)), greedy]
        low_support = chosen_propensity < min_propensity
        low_gap = q_gap < q_gap_threshold
        use_fallback = low_support | low_gap
        final_action = greedy.copy()
        if fallback_to_logged and "action" in df.columns:
            final_action[use_fallback] = df["action"].to_numpy(dtype=int)[use_fallback]
        return pd.DataFrame(
            {
                "policy_action": final_action,
                "greedy_action": greedy,
                "q0": q_values[:, 0],
                "q1": q_values[:, 1],
                "q_gap": q_gap,
                "chosen_propensity": chosen_propensity,
                "low_support": low_support.astype(int),
                "low_gap": low_gap.astype(int),
                "used_fallback": use_fallback.astype(int),
            },
            index=df.index,
        )


def fit_heuristic_risk_model(df: pd.DataFrame, state_columns: list[str]) -> Pipeline:
    preprocessor = build_preprocessor(df, state_columns)
    model = HistGradientBoostingClassifier(
        max_depth=6,
        max_iter=120,
        learning_rate=0.05,
        random_state=SEED,
    )
    late_flag = (df["lateness_min"] > 0).astype(int)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(df[state_columns], late_flag)
    return pipeline


def fit_outcome_models(df: pd.DataFrame, state_columns: list[str]) -> dict[str, Pipeline]:
    models = {}
    augmented_columns = state_columns + [ACTION_COLUMN]
    for offset, target in enumerate(OUTCOME_MODEL_TARGETS):
        pipeline = make_regressor_pipeline(df, augmented_columns, random_state=SEED + offset)
        pipeline.fit(df[augmented_columns], df[target])
        models[target] = pipeline
    return models


def train_all_policies() -> PolicyArtifacts:
    df = load_training_frame()
    metadata = load_metadata()
    train_df = df[df["split"] == "train"].copy()

    causal_state_columns = metadata["causal_state_columns"]
    oracle_state_columns = causal_state_columns + metadata["latent_columns"]

    for column in metadata["latent_columns"]:
        next_col = f"next_state_{column}"
        if next_col not in train_df.columns:
            train_df[next_col] = (
                train_df.groupby(["trajectory_id"], sort=False)[column].shift(-1).fillna(0.0)
            )

    behavior_model = make_behavior_model(train_df, causal_state_columns)
    behavior_model.fit(train_df[causal_state_columns], train_df["action"])

    heuristic_model = fit_heuristic_risk_model(train_df, causal_state_columns)
    outcome_targets = fit_outcome_models(train_df, causal_state_columns)

    learned_policies = {}
    for policy_name, state_columns in LEARNED_POLICY_REGISTRY.items():
        learned_policies[policy_name] = FittedQPolicy(state_columns, name=policy_name).fit(train_df)

    metadata["trained_state_columns"] = {
        **{name: cols for name, cols in LEARNED_POLICY_REGISTRY.items()},
        "oracle_fqi": oracle_state_columns,
    }
    metadata["ablation_definitions"] = {
        "causal_no_history_fqi": {
            "base_policy": "causal_fqi",
            "removed_columns": sorted(set(ABLATION_STATE_REGISTRY["causal_fqi"]) - set(ABLATION_STATE_REGISTRY["causal_no_history_fqi"])),
            "note": "Removes rolling and prior-history features while preserving trajectory position.",
        },
        "causal_no_vehicle_id_fqi": {
            "base_policy": "causal_fqi",
            "removed_columns": sorted(set(ABLATION_STATE_REGISTRY["causal_fqi"]) - set(ABLATION_STATE_REGISTRY["causal_no_vehicle_id_fqi"])),
            "note": "Removes vehicle identity to test whether policy gains depend on vehicle-specific memorization.",
        },
    }
    return PolicyArtifacts(
        behavior_model=behavior_model,
        heuristic_model=heuristic_model,
        outcome_targets=outcome_targets,
        learned_policies=learned_policies,
        metadata=metadata,
    )


def save_policy_artifacts(artifacts: PolicyArtifacts) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts.behavior_model, MODELS_DIR / "behavior_model.joblib")
    joblib.dump(artifacts.heuristic_model, MODELS_DIR / "heuristic_model.joblib")
    joblib.dump(artifacts.outcome_targets, MODELS_DIR / "outcome_targets.joblib")
    for policy_name, policy in artifacts.learned_policies.items():
        joblib.dump(policy, MODELS_DIR / f"{policy_name}.joblib")

    summary = {
        "policies": {},
    }
    for policy_name, policy in artifacts.learned_policies.items():
        summary["policies"][policy_name] = {
            "state_columns": artifacts.metadata["trained_state_columns"][policy_name],
            "training_history_mean_target": policy.training_history,
        }
        if policy_name in artifacts.metadata.get("ablation_definitions", {}):
            summary["policies"][policy_name]["ablation"] = artifacts.metadata["ablation_definitions"][policy_name]
    summary["policies"]["oracle_fqi"] = {
        "state_columns": artifacts.metadata["trained_state_columns"]["oracle_fqi"],
        "note": "Oracle appendix model definition retained in metadata, but excluded from default training for runtime reasons.",
    }
    with POLICY_SUMMARY_PATH.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


def main_results_placeholder() -> None:
    MAIN_RESULTS_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["policy", "expected_reward", "ips", "dr", "fqe"]).to_csv(
        MAIN_RESULTS_TABLE_PATH, index=False
    )
    pd.DataFrame(columns=["reward_name", "policy", "policy_value"]).to_csv(
        REWARD_SENSITIVITY_PATH, index=False
    )
