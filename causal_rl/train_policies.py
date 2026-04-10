from __future__ import annotations

from .policy_learning import main_results_placeholder, save_policy_artifacts, train_all_policies


def main() -> None:
    artifacts = train_all_policies()
    save_policy_artifacts(artifacts)
    main_results_placeholder()
    print("Trained behavior model, heuristic baseline, and FQI policies.")


if __name__ == "__main__":
    main()
