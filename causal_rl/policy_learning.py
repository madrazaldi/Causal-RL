from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
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
    CAUSAL_BACKDOOR_COLUMNS,
    DECISION_LOG_PATH,
    FQI_ITERATIONS,
    GAMMA,
    HEURISTIC_RISK_QUANTILE,
    LEARNED_POLICY_REGISTRY,
    MAIN_RESULTS_TABLE_PATH,
    METADATA_PATH,
    MIN_PROPENSITY,
    MINIMAL_STATE_COLUMNS,
    MODELS_DIR,
    OUTCOME_MODEL_TARGETS,
    POLICY_SUMMARY_PATH,
    Q_GAP_THRESHOLD,
    REWARD_COLUMNS,
    REWARD_SENSITIVITY_PATH,
    SEED,
)
from .parallel import limit_inner_threads, resolve_n_jobs


@dataclass
class PolicyArtifacts:
    behavior_model: Pipeline
    behavior_models: dict[str, Pipeline]
    heuristic_model: Pipeline
    outcome_targets: dict[str, Pipeline]
    outcome_targets_by_state: dict[str, dict[str, Pipeline]]
    learned_policies: dict[str, "FittedQPolicy"]
    metadata: dict
    heuristic_risk_threshold: float


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


    def get_feature_importances(
        self,
        df: pd.DataFrame,
        action: int = 1,
        n_repeats: int = 5,
        random_state: int = SEED,
    ) -> pd.DataFrame:
        from sklearn.inspection import permutation_importance

        if self.preprocessor is None or action not in self.models:
            raise RuntimeError("Policy must be fit before calling get_feature_importances.")
        sample = df.sample(n=min(5000, len(df)), random_state=random_state)
        X = self.preprocessor.transform(sample[self.state_columns])
        y = self.models[action].predict(X)
        result = permutation_importance(
            self.models[action], X, y, n_repeats=n_repeats, random_state=random_state
        )
        raw_names = list(self.preprocessor.get_feature_names_out())
        base_names = []
        for name in raw_names:
            if name.startswith("num__"):
                base_names.append(name[len("num__"):])
            elif name.startswith("cat__"):
                inner = name[len("cat__"):]
                parts = inner.rsplit("_", 1)
                base_names.append(parts[0] if len(parts) == 2 else inner)
            else:
                base_names.append(name)
        importance_df = pd.DataFrame({
            "feature": raw_names,
            "base_feature": base_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        })
        agg = (
            importance_df.groupby("base_feature", as_index=False)
            .agg(importance_mean=("importance_mean", "sum"), importance_std=("importance_std", "mean"))
            .rename(columns={"base_feature": "feature"})
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )
        return agg


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


def calibrate_heuristic_risk_threshold(
    model: Pipeline,
    df: pd.DataFrame,
    state_columns: list[str],
    quantile: float = HEURISTIC_RISK_QUANTILE,
) -> float:
    late_prob = model.predict_proba(df[state_columns])[:, 1]
    return float(np.quantile(late_prob, quantile))


def fit_outcome_models(
    df: pd.DataFrame,
    state_columns: list[str],
    n_jobs: int | str | None = None,
) -> dict[str, Pipeline]:
    augmented_columns = state_columns + [ACTION_COLUMN]

    def _fit_target(offset: int, target: str) -> tuple[str, Pipeline]:
        pipeline = make_regressor_pipeline(df, augmented_columns, random_state=SEED + offset)
        pipeline.fit(df[augmented_columns], df[target])
        return target, pipeline

    jobs = min(resolve_n_jobs(n_jobs), len(OUTCOME_MODEL_TARGETS))
    if jobs == 1:
        fitted = [_fit_target(offset, target) for offset, target in enumerate(OUTCOME_MODEL_TARGETS)]
    else:
        with limit_inner_threads(jobs):
            fitted = Parallel(n_jobs=jobs, prefer="threads")(
                delayed(_fit_target)(offset, target)
                for offset, target in enumerate(OUTCOME_MODEL_TARGETS)
            )
    return dict(fitted)


def train_all_policies(n_jobs: int | str | None = None) -> PolicyArtifacts:
    df = load_training_frame()
    metadata = load_metadata()
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    jobs = resolve_n_jobs(n_jobs)

    causal_state_columns = metadata["causal_state_columns"]
    non_causal_state_columns = metadata["non_causal_state_columns"]
    oracle_state_columns = causal_state_columns + metadata["latent_columns"]

    for column in metadata["latent_columns"]:
        next_col = f"next_state_{column}"
        if next_col not in train_df.columns:
            train_df[next_col] = (
                train_df.groupby(["trajectory_id"], sort=False)[column].shift(-1).fillna(0.0)
            )

    behavior_model = make_behavior_model(train_df, causal_state_columns)
    behavior_model.fit(train_df[causal_state_columns], train_df["action"])
    non_causal_behavior_model = make_behavior_model(train_df, non_causal_state_columns)
    non_causal_behavior_model.fit(train_df[non_causal_state_columns], train_df["action"])
    oracle_behavior_model = make_behavior_model(train_df, oracle_state_columns)
    oracle_behavior_model.fit(train_df[oracle_state_columns], train_df["action"])
    behavior_models = {
        "causal": behavior_model,
        "non_causal": non_causal_behavior_model,
        "oracle": oracle_behavior_model,
    }

    heuristic_model = fit_heuristic_risk_model(train_df, causal_state_columns)
    heuristic_threshold_df = val_df if not val_df.empty else train_df
    heuristic_risk_threshold = calibrate_heuristic_risk_threshold(
        heuristic_model,
        heuristic_threshold_df,
        causal_state_columns,
    )
    outcome_targets = fit_outcome_models(train_df, causal_state_columns, n_jobs=jobs)
    outcome_targets_by_state = {
        "causal": outcome_targets,
        "non_causal": fit_outcome_models(train_df, non_causal_state_columns, n_jobs=jobs),
        "oracle": fit_outcome_models(train_df, oracle_state_columns, n_jobs=jobs),
    }

    def _fit_policy(policy_name: str, state_columns: list[str]) -> tuple[str, FittedQPolicy]:
        policy = FittedQPolicy(state_columns, name=policy_name).fit(train_df)
        return policy_name, policy

    policy_specs = list(LEARNED_POLICY_REGISTRY.items()) + [("oracle_fqi", oracle_state_columns)]
    policy_jobs = min(jobs, len(policy_specs))
    if policy_jobs == 1:
        learned_items = [_fit_policy(policy_name, state_columns) for policy_name, state_columns in policy_specs]
    else:
        with limit_inner_threads(policy_jobs):
            learned_items = Parallel(n_jobs=policy_jobs, prefer="threads")(
                delayed(_fit_policy)(policy_name, state_columns)
                for policy_name, state_columns in policy_specs
            )
    learned_policies = dict(learned_items)

    metadata["trained_state_columns"] = {
        **{name: cols for name, cols in LEARNED_POLICY_REGISTRY.items()},
        "oracle_fqi": oracle_state_columns,
    }
    metadata["oracle_state_columns"] = oracle_state_columns
    metadata["heuristic_policy"] = {
        "risk_quantile": HEURISTIC_RISK_QUANTILE,
        "risk_threshold": heuristic_risk_threshold,
        "threshold_split": "val" if not val_df.empty else "train",
        "note": "The heuristic lateness-risk cutoff is frozen before test evaluation.",
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
        "minimal_fqi": {
            "base_policy": "causal_fqi",
            "removed_columns": sorted(set(CAUSAL_BACKDOOR_COLUMNS) - set(MINIMAL_STATE_COLUMNS)),
            "note": (
                "Minimal 5-feature baseline (hour, demand_size, time_window_tightness, "
                "traffic_index, dispatch_delay_min). Quantifies the value of the full "
                "confounder-aware backdoor-style state over the simplest operationally available features."
            ),
        },
    }
    return PolicyArtifacts(
        behavior_model=behavior_model,
        behavior_models=behavior_models,
        heuristic_model=heuristic_model,
        outcome_targets=outcome_targets,
        outcome_targets_by_state=outcome_targets_by_state,
        learned_policies=learned_policies,
        metadata=metadata,
        heuristic_risk_threshold=heuristic_risk_threshold,
    )


def save_policy_artifacts(artifacts: PolicyArtifacts) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts.behavior_model, MODELS_DIR / "behavior_model.joblib")
    joblib.dump(artifacts.behavior_models, MODELS_DIR / "behavior_models.joblib")
    joblib.dump(artifacts.heuristic_model, MODELS_DIR / "heuristic_model.joblib")
    joblib.dump(artifacts.outcome_targets, MODELS_DIR / "outcome_targets.joblib")
    joblib.dump(artifacts.outcome_targets_by_state, MODELS_DIR / "outcome_targets_by_state.joblib")
    for policy_name, policy in artifacts.learned_policies.items():
        joblib.dump(policy, MODELS_DIR / f"{policy_name}.joblib")

    summary = {
        "policies": {},
        "heuristic_policy": artifacts.metadata["heuristic_policy"],
    }
    for policy_name, policy in artifacts.learned_policies.items():
        summary["policies"][policy_name] = {
            "state_columns": artifacts.metadata["trained_state_columns"][policy_name],
            "training_history_mean_target": policy.training_history,
        }
        if policy_name in artifacts.metadata.get("ablation_definitions", {}):
            summary["policies"][policy_name]["ablation"] = artifacts.metadata["ablation_definitions"][policy_name]
        if policy_name == "oracle_fqi":
            summary["policies"][policy_name]["note"] = (
                "Non-deployable oracle sensitivity model that includes latent simulator columns; "
                "excluded from the main policy table."
            )
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
